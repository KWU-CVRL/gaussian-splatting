#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


 # 1. 모든 카메라의 월드 좌표 위치를 구함
    # 2. 카메라들의 평균 중심점 계산
    # 3. 중심점으로부터 가장 먼 카메라까지의 거리(diagonal) 계산
    # 4. radius = diagonal * 1.1 (약간 여유를 둠)
    # 5. translate = -center (중심을 원점으로 이동)
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):

        # cam_centers = [
                    #                 [[Tx1], [Ty1], [Tz1]],  # camera 1: shape (3, 1)
                    #                 [[Tx2], [Ty2], [Tz2]],  # camera 2: shape (3, 1)
                    #                 [[Tx3], [Ty3], [Tz3]]   # camera 3: shape (3, 1)
                                    #]
        cam_centers = np.hstack(cam_centers) #horizontal stack with N cameras
                    

        #평균을 구하고,
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center

        #각 카메라 위치를 평균에서 빼서 거리를 구함 
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        
        #가장 먼 거리를 계산.
        diagonal = np.max(dist)
    
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:

        # c_T_w; c_T_w * w_p = c_p
        W2C = getWorld2View2(cam.R, cam.T)

        # w_T_c; w_T_c * c_p = w_p
        C2W = np.linalg.inv(W2C)
        
        # [ R  R  R  Tx ]   ← Rotation + Translation
        # [ R  R  R  Ty ]
        # [ R  R  R  Tz ]   ← C2W[:3, 3:4] = [Tx, Ty, Tz]
        # [ 0  0  0   1 ]     C2W[:3, 3:] 도 같음ㅋ
        # 결국 world좌표계 기준으로의 camera마다의 position을 append
        cam_centers.append(C2W[:3, 3:4])
        

    center, diagonal = get_center_and_diag(cam_centers)

    # bounded scene을 가정했기에, 제일 멀리 떨어진 카메라 기준으로 기준으로 10%의 여유를 두고 radius를 설정(구 형태의 scene)
    radius = diagonal * 1.1

    # center를 원점으로 옮기기 위한 translation 벡터, center를 음수로 변환해두기
    # 나중에 더하면 원점 기준으로 옮겨짐
    translate = -center

    return {"translate": translate, "radius": radius}

    # Chummary(Minchu Summary) 1
    # 즉, nerfpp normalization은 scene의 중심을 원점으로 옮기고, radius를 설정하는 것.


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []  # CameraInfo 객체들을 담을 리스트 초기화

    # idx: 현재 카메라 인덱스, key: cam_extrinsics의 키 (카메라 ID)
    # ex)
    # for idx, item in enumerate(['a', 'b', 'c']):
    # print(idx, item)
    # 출력: 
    # 0 a
    # 1 b
    # 2 c

    for idx, key in enumerate(cam_extrinsics):  # 각 카메라에 대해 반복
        sys.stdout.write('\r')  # 커서를 줄 시작으로 이동 (진행상황 덮어쓰기 위함)
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))  # 현재 처리 중인 카메라 번호 출력
        sys.stdout.flush()  # 버퍼 비우고 즉시 출력

        extr = cam_extrinsics[key]  # 현재 카메라의 extrinsic 파라미터 (R, T) 가져오기

        intr = cam_intrinsics[extr.camera_id]  # 해당 카메라의 intrinsic 파라미터 (focal, 해상도) 가져오기
        height = intr.height  # 이미지 높이
        width = intr.width  # 이미지 너비
        uid = intr.id  # 카메라 고유 ID
        
        R = np.transpose(qvec2rotmat(extr.qvec))  # quaternion을 rotation matrix로 변환 후 transpose (COLMAP → OpenGL 좌표계)
        T = np.array(extr.tvec)  # translation vector (카메라 위치)

        # Chummary 2
        # 즉, 어떤 카메라로 촬영했는지 확인하고, 각 카메라마다의 extrinsic, intrinsic 파라미터를 읽어오는 과정 


        if intr.model=="SIMPLE_PINHOLE":  # x, y focal length가 같은 단순 핀홀 모델
            focal_length_x = intr.params[0]  # focal length (단일 값)
            FovY = focal2fov(focal_length_x, height)  # focal → FoV 변환 (세로)
            FovX = focal2fov(focal_length_x, width)  # focal → FoV 변환 (가로)
        elif intr.model=="PINHOLE":  # x, y focal length가 다를 수 있는 일반 핀홀 모델
            focal_length_x = intr.params[0]  # x축 focal length
            focal_length_y = intr.params[1]  # y축 focal length
            FovY = focal2fov(focal_length_y, height)  # y focal로 세로 FoV 계산
            FovX = focal2fov(focal_length_x, width)  # x focal로 가로 FoV 계산
        else:
            # 왜곡 보정되지 않은 카메라 모델은 지원 안 함 → 에러 발생 (distorted images not supported)
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1  # 파일 확장자 길이 + '.' = 제거할 문자 개수 (예: '.jpg' → 4)


        # COLMAP으로부터 이미지마다의 Depth 정보 읽어오기
        depth_params = None  # depth 파라미터 초기화
        if depths_params is not None:  # depth 정보가 제공되었으면
            try:
                depth_params = depths_params[extr.name[:-n_remove]]  # 확장자 제거한 파일명으로 depth params 조회
            except:
                print("\n", key, "not found in depths_params")  # 해당 이미지의 depth 정보 없으면 경고

        #"data/images/photo.jpg"와 같이 image_path와 파일 이름 한줄로 만들어주기 (cross platform 하고싶었대)
        image_path = os.path.join(images_folder, extr.name)  # 이미지 전체 경로 생성
        image_name = extr.name  # 이미지 파일명
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""  # depth 맵 경로 (있으면)

        # 모든 정보를 모아서 CameraInfo 객체 생성
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)  # test set인지 여부 체크
        cam_infos.append(cam_info)  # 리스트에 추가

    sys.stdout.write('\n')  # 진행상황 출력 끝, 줄바꿈
    return cam_infos  
    # Chummary 3 
    # COLMAP으로 추출된 카메라 정보(extrinsics, intrinsics)를 읽어서 CameraInfo 객체 리스트로 변환. 
    # 각 카메라의 R, T, FoV, 이미지 경로, depth 정보 등을 파싱하여 반환

def fetchPly(path):  # path: .ply 파일 경로 (예: "sparse/0/points3D.ply")
    plydata = PlyData.read(path)  # .ply 파일 읽기
    vertices = plydata['vertex']  # vertex 데이터 추출
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # x,y,z 좌표를 (N, 3) 배열로 변환
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # RGB를 (N, 3) 배열로, 0-1 정규화
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # 법선 벡터를 (N, 3) 배열로
    return BasicPointCloud(points=positions, colors=colors, normals=normals)  
    # Chummary 4
    #  .ply 파일로부터 3D 포인트 클라우드 읽어서 BasicPointCloud 객체로 반환 (위치, 색상, 법선 포함)

def storePly(path, xyz, rgb):  # path: 저장할 .ply 파일 경로, xyz: (N,3) 좌표, rgb: (N,3) 색상
    # PLY 파일 포맷을 위한 구조체 정의 (각 vertex의 데이터 타입)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  # 위치: float32
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # 법선: float32
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]  # 색상: uint8
    
    normals = np.zeros_like(xyz)  # 법선 벡터를 0으로 초기화 (COLMAP은 법선 정보 없음)

    elements = np.empty(xyz.shape[0], dtype=dtype)  # 포인트 개수만큼 빈 구조체 배열 생성
    attributes = np.concatenate((xyz, normals, rgb), axis=1)  # (N,3), (N,3), (N,3) → (N,9) 연결
    elements[:] = list(map(tuple, attributes))  # 각 행을 튜플로 변환해서 구조체 배열에 할당
    # map(tuple, attributes) 결과:
    # [(1.0, 2.0, 3.0, 0, 0, 0, 255, 128, 64)]

    # elements[0]에 할당되면:
    # elements[0].x = 1.0
    # elements[0].y = 2.0
    # elements[0].z = 3.0
    # ...
    # elements[0].red = 255


    # PLY 파일 객체 생성 및 저장
    vertex_element = PlyElement.describe(elements, 'vertex')  # vertex element 생성
    ply_data = PlyData([vertex_element])  # PLYData 객체 생성 # tuple 형태로 넣어줘야 함
    ply_data.write(path)  # 파일로 저장
    # Chummary 5
    # 3D 포인트 좌표(xyz)와 색상(rgb)을 받아서 .ply 파일로 저장. 법선은 0으로 초기화

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    # COLMAP 카메라 파라미터 파일 읽기 (binary 우선, 실패하면 text)
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")  # extrinsics binary
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")  # intrinsics binary
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)  # R, T 읽기
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)  # focal, 해상도 읽기
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")  # text 파일로 fallback
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # Depth 파라미터 읽기 (옵션)
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None  # depth 정보 초기화
    if depths != "":  # depth 폴더가 지정되어 있으면
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)  # depth scale 등의 파라미터 로드
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])  # 모든 scale 수집
            if (all_scales > 0).sum():  # 유효한 scale 값이 있으면
                med_scale = np.median(all_scales[all_scales > 0])  # 중앙값 계산
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale  # 각 depth에 중앙값 저장

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)  # depth 필요한데 없으면 에러
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # Test set 분할 (eval 모드일 때만)
    if eval:  # evaluation 모드이면
        if "360" in path:  # 360도 데이터셋이면
            llffhold = 8  # 8개마다 1개씩 test
        if llffhold:  # LLFF hold-out 방식
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]  # 모든 카메라 이름
            cam_names = sorted(cam_names)  # 정렬
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]  # N번째마다 test
        else:  # test.txt 파일이 있으면
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]  # 파일에서 test 이미지 이름 읽기
    else:  # eval 아니면 모두 train
        test_cam_names_list = []

    # 카메라 정보 읽기
    reading_dir = "images" if images == None else images  # 이미지 폴더명 결정
    cam_infos_unsorted = readColmapCameras(  # 카메라 정보 파싱
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)  # 이미지 이름으로 정렬

    # Train/Test 분할
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]  # train_test_exp=True면 모두 train
    test_cam_infos = [c for c in cam_infos if c.is_test]  # test 카메라만 추출

    nerf_normalization = getNerfppNorm(train_cam_infos)  # Chummary 1; 

    # 3D 포인트 클라우드 로드 또는 변환
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):  # .ply 파일이 없으면
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)  # binary에서 포인트 읽기
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)  # text에서 읽기
        storePly(ply_path, xyz, rgb)  # .ply로 변환 저장
    try:
        pcd = fetchPly(ply_path)  # .ply 파일 읽기
    except:
        pcd = None  # 실패하면 None

    # SceneInfo 객체 생성 및 반환
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)  # COLMAP이므로 실제 데이터
    return scene_info
    # Chummary 6
    # COLMAP으로 추출된 데이터셋 전체를 로드: 카메라 정보, 포인트 클라우드, train/test 분할, 씬 정규화까지 모두 처리하여 SceneInfo 반환

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []  # CameraInfo 리스트 초기화, Chummary 3

    with open(os.path.join(path, transformsfile)) as json_file:  # transforms_train.json 또는 transforms_test.json 열기
        contents = json.load(json_file)  # JSON 파싱
        fovx = contents["camera_angle_x"]  # 가로 FoV (라디안)

        frames = contents["frames"]  # 각 프레임(카메라) 정보 리스트
        for idx, frame in enumerate(frames):  # 각 프레임 반복
            cam_name = os.path.join(path, frame["file_path"] + extension)  # 이미지 경로 생성

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])  # C2W 행렬 (4x4)
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1  # Y, Z축 반전 (OpenGL → COLMAP 좌표계)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)  # W2C = inv(C2W)
            R = np.transpose(w2c[:3,:3])  # rotation만 추출 후 transpose (CUDA glm 호환)
            T = w2c[:3, 3]  # translation 추출

            image_path = os.path.join(path, cam_name)  # 전체 이미지 경로
            image_name = Path(cam_name).stem  # 파일명 (확장자 제외)
            image = Image.open(image_path)  # 이미지 로드

            im_data = np.array(image.convert("RGBA"))  # RGBA로 변환 (alpha 채널 포함)

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])  # 배경색 설정

            norm_data = im_data / 255.0  # 0-1 정규화
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])  # alpha blending
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")  # RGB로 변환

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])  # x FoV → focal → y FoV
            FovY = fovy  # 세로 FoV
            FovX = fovx  # 가로 FoV

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""  # depth 경로

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,  # CameraInfo 생성
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos
    # Chummary 7
    # NeRF synthetic 데이터셋의 transforms.json 파일로부터 카메라 정보 읽기. OpenGL 좌표계를 COLMAP으로 변환하고, 
    # alpha blending 처리 후 CameraInfo 리스트 반환
    # Input 자체가 COLMAP이면 의미없다..

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

#ChuStep_1
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
    # Chummary 7은 NeRF synthetic 데이터셋용 읽기 함수 매핑
} 
#즉, 최종 output은 sceneInfo!
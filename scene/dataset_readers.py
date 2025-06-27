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
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pycolmap
import pathlib

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    backward_train_cameras: list 
    backward_test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def compute_scene_bounds(bin_path, margin): 
    try:
        xyz_colmap, _, _ = read_points3D_binary(bin_path)
    except:
        pass 
    if xyz_colmap is not None:
        min_bound = xyz_colmap.min(axis=0)
        max_bound = xyz_colmap.max(axis=0)
        print("Using bounds from COLMAP using cloud", min_bound, ",", max_bound)
    return min_bound - margin, max_bound + margin   

def estimate_focus_center(cam_infos):
    centers = []
    directions = []
    for cam in cam_infos:
        centers.append(cam.T)
        R = cam.R 
        forward = -R[2,:]
        directions.append(forward)
    centers = np.stack(centers)
    directions = np.stack(directions)
    avg_center = centers.mean(axis=0)
    avg_direction = directions.mean(axis=0)
    avg_direction /= np.linalg.norm(avg_direction)
    object_center = avg_center + 1.0 * avg_direction
    print("Estimated center: ", object_center)
    return object_center

def readColmapSceneInfo(path, images, eval, lod, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if lod>0:
            print(f'using lod, using eval')
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0", "points3D.bin")
    low, high = compute_scene_bounds(bin_path, margin=0.5)
    
    num_pts = 10_000
    num_background = 7000
    num_center = 3_000
    print(f"Generating random point cloud ({num_pts})...")
    center = estimate_focus_center(cam_infos)    
    # We create random points
    xyz_center = np.random.normal(loc=center,size=(num_center, 3))
    xyz_background = np.random.uniform(low=low, high=high, size=(num_background, 3))
    xyz = np.concatenate([xyz_center, xyz_background], axis=0)
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
                           ply_path=ply_path)
    return scene_info

# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             im_data = np.array(image.convert("RGBA"))

#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx

#             cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                             image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
#     return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    
    # Since this data set has no colmap data, we start with random points
    num_pts = 10_000
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
                           ply_path=ply_path)
    return scene_info

def createCameraTransforms(path, z_spacing=1, white_background=False, training=True):
    images_folder = os.path.join(path, "images")
    image_files = sorted(os.listdir(images_folder))
    cam_infos = [] 
    backward_cam_infos = []

    progress_bar = tqdm(image_files, desc="Loading dataset")
    ct = 0

    for i, image_name in enumerate(image_files):#progress bar visual
        if i % 10 == 0:
            progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(image_files)}"+Style.RESET_ALL})
            progress_bar.update(10)
        if i == len(image_files) - 1:
            progress_bar.close()
        ct += 1

        image_path = os.path.join(images_folder, image_name)
        image = Image.open(image_path)
        width, height = image.size

        if image.mode == "RGBA": #converts image to right format depending on whether it is RGB or RGBA
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        else:
            image = image.convert("RGB")

        R_forward = np.eye(3) #All cameras face the z axis
        R_backward = np.diag([1, 1, -1])
        T = np.array([0.0, 0.0, i * z_spacing]) #All cameras are at [0,0,z] where z is time
        T_backward = R_backward @ T
        FovX = np.radians(50.0)
        FovY = 2 * np.arctan(np.tan(FovX / 2) * height / width)
        cam_infos.append(CameraInfo(uid=i, R=R_forward, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height))
        backward_cam_infos.append(CameraInfo(uid=i, R=R_backward, T=T_backward, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height))
    return cam_infos, backward_cam_infos

def compute_video_bounds(cam_infos, h=0.2, num_pts=10000):
    camera_z_coordinates = [cam.T[2] for cam in cam_infos] #Work out the min and max z within which to spawn gaussians
    z_min = min(camera_z_coordinates) - h 
    z_max = max(camera_z_coordinates) + h

    y_depth = np.tan(cam_infos[0].FovY / 2) * h #uses the view depth h and the fov to work out how far to distribute gaussians so the whole image fits on the screen and is evenly filled with gaussians
    x_depth = np.tan(cam_infos[0].FovX / 2) * h
    x_min, x_max = -x_depth, x_depth 
    y_min, y_max = -y_depth, y_depth 

    x = np.random.uniform(x_min, x_max, size=num_pts)
    y = np.random.uniform(y_min, y_max, size=num_pts)
    z = np.random.uniform(z_min, z_max, size=num_pts)
    xyz = np.stack([x, y, z], axis=1)
    return xyz

def generate_colmap_gaussians(path):
    dataset_path = pathlib.Path(path)
    image_path = dataset_path / 'images'
    output_path = dataset_path / 'sparse'
    sparse_path = output_path / '0'
    db_path = dataset_path / 'database.db'
    ply_path = sparse_path / 'points3D.ply'

    pycolmap.extract_features(db_path, image_path)
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, image_path, output_path)
    maps[0].write(output_path)

    points_path = sparse_path / 'points3D.bin'

    xyz, rgb, _ = read_points3D_binary(str(points_path)) 
    colors = rgb / 255.0
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    storePly(str(ply_path), xyz, rgb)
    return pcd, str(ply_path)

def readVideoInfo(path, white_background, eval, ply_path, training):
    #create_colmap()
    #return
    z_spacing = 0.1
    print("Generating Training Transforms")
    train_cam_infos, backward_train_cam_infos = createCameraTransforms(path, z_spacing=z_spacing, white_background=white_background, training=True)
    print("Generating Test Transforms")
    test_cam_infos, backward_test_cam_infos =  createCameraTransforms(path, z_spacing=z_spacing, white_background=white_background, training=False)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        ply_path_alt = os.path.join(path, "points3D.ply")
    
    if not os.path.exists(ply_path):
        ply_path = ply_path_alt
        if  not os.path.exists(ply_path) or training: 
            num_pts = 5_000
            h = 0.05
            print(f"Generating random point cloud ({num_pts})...")

            xyz = compute_video_bounds(train_cam_infos, h, num_pts)
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
        else: 
            print("Found existing gaussian cloud")
    else:
        print("Pre-existing Colmap Found")

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            backward_train_cameras=backward_train_cam_infos,
                            backward_test_cameras=backward_test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Video": readVideoInfo
}

import shutil

def downsampled_subset(images_dir: str, output_dir: str, step: int = 30) -> str:
    images_dir = pathlib.Path(images_dir)
    output_dir = pathlib.Path(output_dir)

    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    for i, img_path in enumerate(image_files):
        if i % step == 0:
            shutil.copy(img_path, output_dir / img_path.name)
    return str(output_dir)

def create_colmap():
    dataset_path = pathlib.Path("data/videos/ShakeNDry")
    pycolmap.verbose=True
    image_path = dataset_path / 'images'
    subset_path = 'images'
    output_path = dataset_path / 'sparse'
    sparse_path = output_path / '0'
    db_path = dataset_path / 'database.db'
    ply_path = sparse_path / 'points3D.ply'

    downsampled_path = downsampled_subset(image_path, subset_path, step=8)

    pycolmap.extract_features(db_path, downsampled_path)
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, downsampled_path, output_path)
    maps[0].write(output_path)

    points_path = sparse_path / 'points3D.bin'

    xyz, rgb, _ = read_points3D_binary(str(points_path)) 
    
    storePly(str(ply_path), xyz, rgb)
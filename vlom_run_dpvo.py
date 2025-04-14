import sys
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.utils import Timer


def load_poses(path):
    c2ws = np.load(path)["poses"]
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    return c2ws


def scannet_image_stream(queue, scene_dir):
    """ Image generator for ScanNet """
    imagedir = Path(scene_dir)
    video_path = imagedir / "video.mp4"
    pose_path = imagedir / "poses.npz"
    intrinsics_path = imagedir / "intrinsics.npz"

    vr = VideoReader(str(video_path))
    c2ws = np.load(pose_path)["poses"]

    # Take everything until first invalid pose
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]

    # Move to the origin for visualization
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws

    intrinsics = np.load(intrinsics_path)["poses"]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    intrinsics = np.array([fx, fy, cx, cy])

    for i in range(c2ws.shape[0]):
        image = vr[i].asnumpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        queue.put((float(i), image, c2ws[i], intrinsics))

    queue.put((-1, image, c2ws[i], intrinsics))


def arkit_image_stream(queue, imagedir):
    """ Image generator for ScanNet """
    imagedir = Path(imagedir)
    video_path = imagedir / "video.mp4"
    pose_path = imagedir / "poses.npz"
    intrinsics_path = imagedir / "intrinsics.npz"

    vr = VideoReader(str(video_path))
    c2ws = np.load(pose_path)["poses"]

    # Take everything until first invalid pose
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]

    # Move to the origin for visualization
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws

    # Load intrinsics
    all_intrinsics = np.load(intrinsics_path)["intrinsics"]

    for i in range(c2ws.shape[0]):
        image = vr[i].asnumpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        intrinsics = all_intrinsics[i]
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        intrinsics = np.array([fx, fy, cx, cy])
        queue.put((float(i), image, c2ws[i], intrinsics))
    queue.put((-1, image, c2ws[i], intrinsics))


def tum_image_stream(queue, scene_dir):
    """ Image generator for TUM RGB-D dataset with undistortion """
    scene_dir = Path(scene_dir)

    # Load RGB and pose data
    rgb_txt = np.loadtxt(scene_dir / "rgb.txt", dtype=str, skiprows=1)
    pose_txt = np.loadtxt(scene_dir / "groundtruth.txt", dtype=str, skiprows=1)

    rgb_timestamps = rgb_txt[:, 0].astype(np.float64)
    rgb_files = rgb_txt[:, 1]
    pose_timestamps = pose_txt[:, 0].astype(np.float64)
    pose_vecs = pose_txt[:, 1:].astype(np.float64)

    # Associate frames and poses
    max_dt = 0.08
    associations = []
    for i, t in enumerate(rgb_timestamps):
        k = np.argmin(np.abs(pose_timestamps - t))
        if np.abs(pose_timestamps[k] - t) < max_dt:
            associations.append((i, k))

    # Convert to 4x4 pose matrices
    from scipy.spatial.transform import Rotation
    poses = []
    for _, k in associations:
        pvec = pose_vecs[k]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        poses.append(pose)
    poses = np.stack(poses, axis=0)

    # Normalize poses to origin
    poses = np.linalg.inv(poses[0]) @ poses

    # Intrinsics and distortion coefficients for freiburg1
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    # Apply undistortion, crop, and push to queue
    for i, (img_idx, _) in enumerate(associations):
        img_path = scene_dir / rgb_files[img_idx]
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # skip unreadable files

        image = cv2.undistort(image, K_l, d_l)
        image = image[8:-8, 16:-16]  # Crop H and W
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.transpose(2, 0, 1)  # CHW

        # Adjust intrinsics to account for cropping
        intrinsics = np.array([fx, fy, cx - 16, cy - 8])

        queue.put((float(i), image, poses[i], intrinsics))

    # End-of-stream marker
    queue.put((-1, image, poses[-1], intrinsics))


@torch.no_grad()
def run(cfg, network, scene_dir):
    """ Runs SLAM and saves results to output_dir """
    slam = None

    queue = Queue(maxsize=8)
    if "ScanNet" in str(scene_dir):
        reader = Process(target=scannet_image_stream, args=(queue, scene_dir))
    elif "tum" in str(scene_dir):
        reader = Process(target=tum_image_stream, args=(queue, scene_dir))
    else:
        reader = Process(target=arkit_image_stream, args=(queue, scene_dir))
    reader.start()

    gt_poses = []
    for _ in range(sys.maxsize):
        (t, images, gt_pose, intrinsics) = queue.get()
        if t < 0:
            break

        images = torch.as_tensor(images, device='cuda')
        intrinsics = torch.as_tensor(intrinsics, device='cuda')
        gt_poses.append(gt_pose)

        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[-2], wd=images.shape[-1], viz=False)

        intrinsics = intrinsics.cuda()
        with Timer("SLAM", enabled=False):
            slam(t, images, intrinsics)

    reader.join()

    poses, timestamps = slam.terminate()

    return poses, np.array(gt_poses), timestamps


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--max_seq_length', type=int, default=-1)
    parser.add_argument('--scannet_dir', type=Path, default="datasets/ScanNetVideos")
    parser.add_argument('--split_file', type=Path, required=True)
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--output_path', type=Path, required=True)  # New output directory argument
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1)

    # Load scene names from the split file
    with open(args.split_file, 'r') as f:
        scannet_scenes = sorted([line.strip() for line in f.readlines()])

    results = {}
    for i, scene in enumerate(scannet_scenes):
        print("Processing", scene, i + 1, "/", len(scannet_scenes))
        scene_dir = args.scannet_dir / f"{scene}"
        output_scene_dir = args.output_path / scene

        traj_est, gt_poses, timestamps = run(cfg, args.network, scene_dir)

        output_scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_scene_dir / "pred_traj.npy", traj_est)
        np.save(output_scene_dir / "gt_traj.npy", gt_poses)

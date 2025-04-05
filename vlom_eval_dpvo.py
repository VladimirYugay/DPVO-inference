import argparse
import io
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import evo
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import PlotMode
from PIL import Image
from scipy.spatial.transform import Rotation


def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align)
    ate = ate_result.stats["rmse"]

    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align)
    are = are_result.stats["rmse"]

    # RPE rotation and translation
    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_rot = rpe_rots_result.stats["rmse"]

    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_trans = rpe_transs_result.stats["rmse"]

    plot_mode = PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est,
        ate_result.np_arrays["error_array"],
        plot_mode,
        min_map=ate_result.stats["min"],
        max_map=ate_result.stats["max"],
    )
    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return {
        "ate": ate,
        "are": are,
        "rpe_rot": rpe_rot,
        "rpe_trans": rpe_trans
    }, pillow_image


def load_tum_trajectory(filepath):
    data = np.load(filepath)
    positions = data[:, :3]                     # (N, 3)
    quats_xyzw = data[:, 3:7]                   # qx, qy, qz, qw

    rotations = Rotation.from_quat(quats_xyzw)          # (N,) Rotation objects
    matrices = rotations.as_matrix()             # (N, 3, 3)

    poses = np.eye(4)[None].repeat(len(data), axis=0)  # (N, 4, 4)
    poses[:, :3, :3] = matrices
    poses[:, :3, 3] = positions
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to the folder containing predictions")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder containing predictions")
    parser.add_argument("--plot_traj", action="store_true", help="Plot trajectory")
    parser.add_argument("--eval_stride", default=1, type=int, help="Stride for evaluation only")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    average_metrics = defaultdict(list)

    predictions_path = Path(args.predictions_path)
    scene_names = [scene_path.parts[-1] for scene_path in sorted(predictions_path.glob("*")) if scene_path.is_dir()]
    for i, scene_name in enumerate(scene_names):
        print(f"Processing scene: {scene_name}, {i + 1}/{len(scene_names)}")
        pred_poses = load_tum_trajectory(predictions_path / scene_name / "pred_traj.npy")
        gt_poses = np.load(predictions_path / scene_name / "gt_traj.npy")

        # Evaluate every stride frame to match the eval protocol of Cut3r and VGGT
        pred_poses = pred_poses[::args.eval_stride]
        gt_poses = gt_poses[::args.eval_stride]

        metrics, traj_img = eval_trajectory(pred_poses, gt_poses, np.arange(len(pred_poses)), align=False)
        aligned_metrics, _ = eval_trajectory(pred_poses, gt_poses, np.arange(len(pred_poses)), align=True)

        all_metrics = deepcopy(metrics)
        for key in aligned_metrics:
            all_metrics[f"aligned_{key}"] = aligned_metrics[key]

        (output_path / scene_name).mkdir(parents=True, exist_ok=True)
        with open(output_path / scene_name / "metrics.json", "w") as file:
            json.dump(metrics, file)

        if args.plot_traj:
            traj_img.save(output_path / scene_name / "plot.png")

        for key, value in all_metrics.items():
            average_metrics[key].append(value)

    for key in average_metrics.keys():
        average_metrics[key] = np.mean(average_metrics[key])

    with open(output_path / "average_results.json", "w") as f:
        json.dump(average_metrics, f, indent=4)


if __name__ == "__main__":
    main()

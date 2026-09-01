"""Motion resampling and compact 3D interaction visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from utils import paramUtil


_ACTOR_PALETTES = (
    ("coral", "coral", "orangered", "coral", "coral"),
    ("royalblue", "royalblue", "navy", "royalblue", "royalblue"),
)
_TWO_VIEW_COLORS = (
    "orange",
    "green",
    "black",
    "red",
    "blue",
    "darkblue",
    "darkred",
)
_FLOOR_COLOR = (0.5, 0.5, 0.5, 0.5)


@dataclass
class _MotionTrack:
    joints: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    trajectory: np.ndarray
    rotated: np.ndarray | None = None


def preprocess_plot_motion(
    motions, caption, vis_dir, npy_dir, file_name, foot_ik=False
):
    """Save both actors' motion arrays and render their joint animation."""
    motion_array = np.asarray(motions)
    if motion_array.ndim != 3 or motion_array.shape[1] < 2:
        raise ValueError(
            "motions must have shape (frames, actors, features) with two actors"
        )
    if motion_array.shape[2] < 66:
        raise ValueError("motions must contain at least 22 three-dimensional joints")

    npy_root = Path(npy_dir)
    sequences = []
    for actor_index in range(2):
        actor_motion = motion_array[:, actor_index]
        joints = actor_motion[:, :66].reshape(-1, 22, 3)
        sequences.append(gaussian_filter1d(joints, sigma=1, axis=0, mode="nearest"))
        np.save(npy_root / f"{file_name}_{actor_index}.npy", actor_motion)

    video_root = Path(vis_dir)
    plot_3d_motion(
        video_root / f"{file_name}.mp4",
        paramUtil.t2m_kinematic_chain,
        sequences,
        title=caption,
        fps=30,
    )
    if foot_ik:
        raise NotImplementedError(
            "foot_ik visualization requires an IK sequence, which is not "
            "produced by this preprocessing function"
        )


def list_cut_average(values, intervals):
    """Average consecutive chunks while retaining a final partial chunk."""
    if intervals <= 0:
        raise ValueError("intervals must be positive")
    if intervals == 1:
        return values
    return [
        np.mean(values[start : start + intervals])
        for start in range(0, len(values), intervals)
    ]


def resample_motion_linear(
    motion: np.ndarray, src_fps: int = 20, dst_fps: int = 30
) -> np.ndarray:
    """Linearly resample every motion feature while preserving duration."""
    samples = np.asarray(motion)
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be positive")
    if samples.ndim < 2:
        raise ValueError("motion must have a frame axis and at least one feature axis")
    if samples.shape[0] <= 1 or src_fps == dst_fps:
        return samples

    source_times = np.arange(samples.shape[0], dtype=np.float64) / src_fps
    destination_size = round(source_times[-1] * dst_fps) + 1
    destination_times = np.linspace(
        0.0, source_times[-1], destination_size, dtype=np.float64
    )

    flattened = samples.reshape(samples.shape[0], -1)
    interpolated = np.stack(
        [
            np.interp(destination_times, source_times, flattened[:, feature])
            for feature in range(flattened.shape[1])
        ],
        axis=1,
    )
    output_shape = (destination_size,) + samples.shape[1:]
    return interpolated.reshape(output_shape).astype(samples.dtype, copy=False)


def _wrapped_title(title: str, words_per_line: int = 10) -> str:
    words = title.split()
    return "\n".join(
        " ".join(words[start : start + words_per_line])
        for start in range(0, len(words), words_per_line)
    )


def _motion_track(joints, rotation: Rotation | None = None) -> _MotionTrack:
    data = np.asarray(joints).copy().reshape(len(joints), -1, 3)
    if data.shape[0] == 0:
        raise ValueError("motion sequences cannot be empty")

    bounds_min = data.min(axis=(0, 1))
    bounds_max = data.max(axis=(0, 1))
    data[:, :, 1] -= bounds_min[1]
    rotated = None
    if rotation is not None:
        rotated = rotation.apply(data.reshape(-1, 3)).reshape(data.shape)
    return _MotionTrack(
        joints=data,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        trajectory=data[:, 0, (0, 2)],
        rotated=rotated,
    )


def _prepare_tracks(motions, rotation: Rotation | None = None):
    if len(motions) == 0:
        raise ValueError("mp_joints must contain at least one motion")
    tracks = [_motion_track(motion, rotation) for motion in motions]
    frame_count = min(track.joints.shape[0] for track in tracks)
    return tracks, frame_count


def _add_floor(axis, x_min, x_max, height, z_min, z_max):
    vertices = [
        (x_min, height, z_min),
        (x_min, height, z_max),
        (x_max, height, z_max),
        (x_max, height, z_min),
    ]
    floor = Poly3DCollection([vertices], facecolors=[_FLOOR_COLOR])
    axis.add_collection3d(floor)


def _style_axis(axis, x_limits, y_limits, z_limits, distance):
    axis.set_xlim3d(x_limits)
    axis.set_ylim3d(y_limits)
    axis.set_zlim3d(z_limits)
    axis.view_init(elev=120, azim=-90)
    axis.dist = distance
    axis.grid(False)
    axis.set_axis_off()


def _chain_color(palette: Sequence[str], chain_index: int) -> str:
    return palette[min(chain_index, len(palette) - 1)]


def _draw_skeleton(axis, joints, kinematic_tree, palette):
    for chain_index, chain in enumerate(kinematic_tree):
        axis.plot3D(
            joints[chain, 0],
            joints[chain, 1],
            joints[chain, 2],
            linewidth=4.0 if chain_index < 5 else 2.0,
            color=_chain_color(palette, chain_index),
            alpha=1.0,
        )


def _save_animation(figure, update, frame_count, save_path, fps):
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=False,
    )
    try:
        animation.save(str(save_path), fps=fps)
    finally:
        plt.close(figure)


def plot_3d_motion(
    save_path,
    kinematic_tree,
    mp_joints,
    title,
    figsize=(10, 10),
    fps=120,
    radius=3,
    follow_trajec=False,
):
    """Render interacting skeletons from a single elevated camera."""
    tracks, frame_count = _prepare_tracks(mp_joints)
    figure = plt.figure(figsize=figsize)
    axis = figure.add_subplot(111, projection="3d")
    figure.suptitle(_wrapped_title(title), fontsize=20)

    def update(frame_index):
        axis.cla()
        camera_distance = 7.5 if follow_trajec else 15
        _style_axis(
            axis,
            (-radius / 2, radius / 2),
            (0, radius),
            (-radius / 3, radius * 2 / 3),
            camera_distance,
        )

        focus_x = tracks[0].trajectory[frame_index, 0] if follow_trajec else 0.0
        focus_z = tracks[0].trajectory[frame_index, 1] if follow_trajec else 0.0
        if follow_trajec:
            _add_floor(
                axis,
                tracks[0].bounds_min[0] - focus_x,
                tracks[0].bounds_max[0] - focus_x,
                0,
                tracks[0].bounds_min[2] - focus_z,
                tracks[0].bounds_max[2] - focus_z,
            )
        else:
            _add_floor(axis, -3, 3, 0, -3, 3)

        actors_by_depth = []
        for actor_index, track in enumerate(tracks):
            frame = track.joints[frame_index].copy()
            if follow_trajec:
                frame[:, 0] -= focus_x
                frame[:, 2] -= focus_z
            actors_by_depth.append((frame[:, 2].mean(), actor_index, frame))

        for _, actor_index, frame in sorted(actors_by_depth):
            palette = _ACTOR_PALETTES[actor_index % len(_ACTOR_PALETTES)]
            _draw_skeleton(axis, frame, kinematic_tree, palette)

    _save_animation(figure, update, frame_count, save_path, fps)


def plot_3d_motion_2views(
    save_path,
    kinematic_tree,
    mp_joints,
    title,
    figsize=(20, 10),
    fps=120,
    radius=8,
    foots=None,
):
    """Render the same interaction from world and rotated viewpoints."""
    rotation = Rotation.from_euler("y", 110, degrees=True)
    tracks, frame_count = _prepare_tracks(mp_joints, rotation=rotation)
    figure = plt.figure(figsize=figsize)
    axes = (
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    )
    figure.suptitle(_wrapped_title(title), fontsize=20)

    def update(frame_index):
        for axis in axes:
            axis.cla()
            _style_axis(
                axis,
                (-radius / 4, radius / 4),
                (0, radius / 4),
                (0, radius / 4),
                15,
            )
            axis.view_init(elev=120, azim=270)
            _add_floor(axis, -3, 3, 0, -3, 3)

        for actor_index, track in enumerate(tracks):
            color = _TWO_VIEW_COLORS[actor_index % len(_TWO_VIEW_COLORS)]
            if foots is not None:
                _, left_toe, _, _ = foots[actor_index][:, frame_index]
                if left_toe == 1:
                    color = "darkred"
            palette = (color,) * len(kinematic_tree)
            _draw_skeleton(axes[0], track.joints[frame_index], kinematic_tree, palette)
            _draw_skeleton(axes[1], track.rotated[frame_index], kinematic_tree, palette)

    _save_animation(figure, update, frame_count, save_path, fps)

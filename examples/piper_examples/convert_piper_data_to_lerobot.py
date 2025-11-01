# examples/piper_examples/convert_piper_data_to_lerobot.py
"""
Convert an ETHRC Piper dataset that is already in LeRobot (parquet/mp4) format
into a Pi05-friendly LeRobot dataset with the keys expected by OpenPI.
"""


from __future__ import annotations
import shutil
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import tyro
from tqdm.auto import tqdm

from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

# Candidate keys in the source dataset for each modality we need.
CAMERA_KEY_CANDIDATES: Mapping[str, Sequence[str]] = {
    "observation.images.stereo": (
        "observation.images.stereo",
        "observation.images.main",
    ),
    "observation.images.wrist1": (
        "observation.images.wrist1",
        "observation.images.left_wrist",
    ),
    "observation.images.wrist2": (
        "observation.images.wrist2",
        "observation.images.right_wrist",
    ),
}

STATE_KEY_CANDIDATES = (
    "observation.state",
    "observation.qpos",
)

ACTION_KEY_CANDIDATES = (
    "actions",
    "action",
)

REWARD_KEY_CANDIDATES = (
    "reward",
    "rewards",
    "observation.reward",
)


@dataclass
class Args:
    source_repo_id: str = "ETHRC/piper_towel_v0_with_rewards"
    target_repo_id: str = "your_hf_username/piper_towel_v0_lerobot"
    split: str = "train"
    max_episodes: int | None = None
    download_videos: bool = True
    overwrite: bool = False
    push_to_hub: bool = False
    push_branch: str | None = None
    push_tags: List[str] = dataclasses.field(
        default_factory=lambda: ["piper", "two-arm", "manipulation"]
    )
    push_license: str = "apache-2.0"
    push_private: bool = False
    push_without_videos: bool = False


def _parse_split(meta: LeRobotDatasetMetadata, split: str) -> List[int]:
    if split not in meta.info["splits"]:
        raise ValueError(f"Split '{split}' not found in dataset splits: {list(meta.info['splits'])}")
    split_spec = meta.info["splits"][split]
    if isinstance(split_spec, str):
        split_spec = [split_spec]
    indices: List[int] = []
    for chunk in split_spec:
        if isinstance(chunk, str):
            start, end = chunk.split(":")
            indices.extend(range(int(start), int(end)))
        elif isinstance(chunk, dict):
            indices.extend(range(int(chunk["start"]), int(chunk["end"])))
        else:
            raise ValueError(f"Unsupported split specification: {chunk}")
    return indices


def _resolve_first_present(candidates: Iterable[str], features: Mapping[str, Dict]) -> str:
    for key in candidates:
        if key in features:
            return key
    raise KeyError(f"None of the keys {list(candidates)} were found in the source dataset.")


def _resolve_camera_map(features: Mapping[str, Dict]) -> Dict[str, str]:
    camera_map: Dict[str, str] = {}
    for target_key, candidates in CAMERA_KEY_CANDIDATES.items():
        camera_map[target_key] = _resolve_first_present(candidates, features)
    return camera_map


def _resolve_scalar_key(candidates: Sequence[str], features: Mapping[str, Dict]) -> str:
    return _resolve_first_present(candidates, features)


def _maybe_resolve_scalar_key(
    candidates: Sequence[str], features: Mapping[str, Dict]
) -> str | None:
    try:
        return _resolve_first_present(candidates, features)
    except KeyError:
        return None


def _copy_feature_definition(source: Dict) -> Dict:
    return {key: value for key, value in source.items()}


def _to_float(value) -> float:
    if isinstance(value, np.ndarray):
        return float(value.item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def main(args: Args) -> None:
    source_meta = LeRobotDatasetMetadata(
        repo_id=args.source_repo_id,
        force_cache_sync=True,
    )
    episode_indices = _parse_split(source_meta, args.split)
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]
    if not episode_indices:
        raise ValueError("No episodes selected for conversion.")

    source_dataset = LeRobotDataset(
        repo_id=args.source_repo_id,
        episodes=episode_indices,
        download_videos=args.download_videos,
    )
    source_features = source_dataset.meta.features

    camera_map = _resolve_camera_map(source_features)
    state_key = _resolve_scalar_key(STATE_KEY_CANDIDATES, source_features)
    action_key = _resolve_scalar_key(ACTION_KEY_CANDIDATES, source_features)
    reward_key = _maybe_resolve_scalar_key(REWARD_KEY_CANDIDATES, source_features)

    target_features: Dict[str, Dict] = {}
    for target_key, source_key in camera_map.items():
        target_features[target_key] = _copy_feature_definition(source_features[source_key])
        cam_ft = target_features[target_key]
        if tuple(cam_ft.get("names", [])) == ("channels", "height", "width"):
            channels, height, width = cam_ft["shape"]
            cam_ft["shape"] = (height, width, channels)
            cam_ft["names"] = ["height", "width", "channel"]
            cam_ft["dtype"] = "image"

    target_features["observation.state"] = _copy_feature_definition(source_features[state_key])
    target_features["actions"] = _copy_feature_definition(source_features[action_key])
    if reward_key is not None:
        target_features["reward"] = _copy_feature_definition(source_features[reward_key])

    target_root = HF_LEROBOT_HOME / Path(args.target_repo_id)
    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{target_root} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(target_root)

    target_dataset = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        fps=source_dataset.meta.fps,
        robot_type=source_dataset.meta.robot_type,
        features=target_features,
        use_videos=True,
    )
    target_dataset.meta.info["features"]["timestamp"]["shape"] = ()
    target_dataset.meta.features["timestamp"]["shape"] = ()

    if "reward" in target_dataset.meta.features:
        target_dataset.meta.info["features"]["reward"]["shape"] = ()
        target_dataset.meta.features["reward"]["shape"] = ()

    selected_episodes = set(episode_indices)
    current_episode = None
    processed_frames = 0

    for sample in tqdm(source_dataset, desc="Converting frames"):
        ep_idx = int(_to_float(sample["episode_index"]))
        if ep_idx not in selected_episodes:
            continue

        if current_episode is None:
            current_episode = ep_idx
        elif ep_idx != current_episode:
            target_dataset.save_episode()
            target_dataset.clear_episode_buffer()
            current_episode = ep_idx

        frame_payload: Dict[str, object] = {}
        timestamp_value = sample.get("timestamp")
        if timestamp_value is not None:
            frame_payload["timestamp"] = np.array(_to_float(timestamp_value), dtype=np.float32)

        if reward_key is not None and reward_key in sample:
            frame_payload["reward"] = np.array(sample[reward_key], dtype=np.float32)

        for target_key, source_key in camera_map.items():
            cam_array = np.asarray(sample[source_key])
            if cam_array.ndim == 3 and cam_array.shape[0] in (1, 3, 4):
                cam_array = np.moveaxis(cam_array, 0, -1)
            cam_array = cam_array.astype(np.uint8, copy=False)
            frame_payload[target_key] = cam_array
 
        frame_payload["observation.state"] = sample[state_key]
        frame_payload["actions"] = sample[action_key]

       

        task_value = sample.get("task")
        if task_value is not None:
            if isinstance(task_value, bytes):
                task_value = task_value.decode("utf-8")
            frame_payload["task"] = str(task_value)

        target_dataset.add_frame(frame_payload)
        processed_frames += 1

    if target_dataset.episode_buffer is not None and target_dataset.episode_buffer.get("size", 0):
        target_dataset.save_episode()
    target_dataset.stop_image_writer()

    print(
        f"Saved converted dataset to {target_root} "
        f"({target_dataset.meta.total_episodes} episodes, {processed_frames} frames)."
    )

    if args.push_to_hub:
        target_dataset.push_to_hub(
            branch=args.push_branch,
            tags=args.push_tags,
            license=args.push_license,
            private=args.push_private,
            push_videos=not args.push_without_videos,
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.target_repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
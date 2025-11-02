import dataclasses
import numpy as np
import einops
from typing import Dict, Any, Optional
from openpi import transforms
from openpi.models import model as _model


def _parse_image(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    Universal adapter for ETHRC Piper datasets.
    Just flip `two_arms=True` to enable both wrists and 14-DoF actions.
    """

    two_arms: bool = True
    dof_per_arm: int = 7
    shared_dof: int = 0
    model_type: _model.ModelType = _model.ModelType.PI0

    # dataset key map
    base_image_key: str = "observation/images/stereo"
    wrist1_key: str = "observation/images/wrist1"
    wrist2_key: str = "observation/images/wrist2"
    state_key: Optional[str] = None
    actions_key: Optional[str] = "actions"
    prompt_key: Optional[str] = "language_instruction"

    @property
    def total_action_dim(self) -> int:
        if self.two_arms:
            return self.shared_dof + 2 * self.dof_per_arm
        else:
            return self.shared_dof + self.dof_per_arm

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        mask_padding = self.model_type == _model.ModelType.PI0

        # ----- proprio / state -----
        if self.state_key and self.state_key in data:
            state = np.asarray(data[self.state_key])
        else:
            state = np.zeros(self.total_action_dim, dtype=np.float32)
        state = transforms.pad_to_dim(state, self.total_action_dim)

        # ----- images -----
        base = _parse_image(data[self.base_image_key])
        wrist1 = _parse_image(data[self.wrist1_key]) if self.wrist1_key in data else np.zeros_like(base)
        wrist2 = (
            _parse_image(data[self.wrist2_key])
            if self.two_arms and self.wrist2_key in data
            else np.zeros_like(base)
        )

        images = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": wrist1,
        }
        image_mask = {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_}

        if self.two_arms:
            images["right_wrist_0_rgb"] = wrist2
            image_mask["right_wrist_0_rgb"] = np.True_
        else:
            # fill missing right wrist with zeros if PI0 expects all slots
            images["right_wrist_0_rgb"] = np.zeros_like(base)
            image_mask["right_wrist_0_rgb"] = np.False_ if mask_padding else np.True_

        inputs = {"state": state, "image": images, "image_mask": image_mask}

        # ----- actions (training only) -----
        if self.actions_key and self.actions_key in data:
            acts = transforms.pad_to_dim(np.asarray(data[self.actions_key]),
                                         self.total_action_dim)
            inputs["actions"] = acts

        # ----- language -----
        if self.prompt_key and self.prompt_key in data:
            inputs["prompt"] = data[self.prompt_key]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    two_arms: bool = True
    dof_per_arm: int = 7
    shared_dof: int = 0

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        acts = np.asarray(data["actions"])
        total = self.shared_dof + (2 if self.two_arms else 1) * self.dof_per_arm
        acts = acts[:, :total]

        off = 0
        out = {}
        if self.shared_dof:
            out["actions_shared"] = acts[:, off:off+self.shared_dof]
            off += self.shared_dof

        out["actions_arm0"] = acts[:, off:off+self.dof_per_arm]
        off += self.dof_per_arm

        if self.two_arms:
            out["actions_arm1"] = acts[:, off:off+self.dof_per_arm]
            off += self.dof_per_arm

        out["actions"] = acts
        return out

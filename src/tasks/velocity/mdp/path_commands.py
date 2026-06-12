from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def constant_path_speed_command(
    env: "ManagerBasedRlEnv",
    desired_speed: float = 0.4,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """High-level path speed command.

    Output:
      [desired_path_speed / speed_scale]
    """
    speed_scale = max(float(speed_scale), 1e-6)
    cmd = torch.full(
        (env.num_envs, 1),
        float(desired_speed) / speed_scale,
        device=env.device,
    )
    return cmd
import torch


def get_current_bridge_half_width(
    env,
    max_bridge_half_width: float = 0.8,
    min_bridge_half_width: float = 0.3,
) -> torch.Tensor:
    """Return current bridge half width for each env based on terrain curriculum level."""

    # fallback: no terrain curriculum
    width = torch.full(
        (env.num_envs,),
        max_bridge_half_width,
        device=env.device,
        dtype=torch.float32,
    )

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return width

    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is None:
        return width

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return width

    num_rows = getattr(terrain_generator, "num_rows", 1)
    if num_rows <= 1:
        return width

    difficulty = terrain_levels.float() / float(num_rows - 1)
    difficulty = torch.clamp(difficulty, 0.0, 1.0)

    width = (
        max_bridge_half_width
        - difficulty * (max_bridge_half_width - min_bridge_half_width)
    )

    return torch.clamp(width, min=min_bridge_half_width)
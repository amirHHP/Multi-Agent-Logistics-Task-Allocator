"""Render a short GIF of a trained policy (or random) on the logistics env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.logistics_multi_env import parallel_env  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="", help="Path to PPO zip (without .zip ok)")
    p.add_argument("--output", type=str, default=str(ROOT / "results" / "demo.gif"))
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("Install imageio: pip install imageio")

    from stable_baselines3 import PPO

    env = parallel_env(num_agents=3, render_mode="rgb_array", max_steps=200)
    obs, _ = env.reset(seed=args.seed)

    model = None
    if args.model:
        path = Path(args.model)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        model = PPO.load(str(path))

    frames = []
    for _ in range(args.frames):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if model is None:
            actions = {a: env.action_space(a).sample() for a in env.agents}
        else:
            actions = {
                a: int(model.predict(obs[a], deterministic=True)[0])
                for a in env.agents
            }
        obs, _, term, trunc, _ = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            break

    env.close()
    if not frames:
        raise SystemExit("No frames captured; check render_mode.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, duration=0.08)
    print(f"Saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()

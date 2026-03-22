"""
Train a shared PPO policy on the logistics ParallelEnv via SuperSuit vectorization.

Follows the PettingZoo + SB3 pattern (single policy reused for every agent).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.logistics_multi_env import parallel_env  # noqa: E402


def build_vec_env(
    num_agents: int,
    seed: int,
    n_vec: int,
    num_cpus: int,
) -> Any:
    env = parallel_env(num_agents=num_agents, render_mode=None)
    env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, n_vec, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    return env


def evaluate_policy(
    num_agents: int,
    model: Optional[Any],
    episodes: int,
    seed: int,
    deterministic: bool,
) -> Dict[str, float]:
    env = parallel_env(num_agents=num_agents, render_mode=None)
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    successes: List[float] = []
    collisions: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep_ret = 0.0
        ep_coll = 0.0
        while True:
            if model is None:
                actions = {a: env.action_space(a).sample() for a in env.agents}
            else:
                actions = {
                    a: int(model.predict(obs[a], deterministic=deterministic)[0])
                    for a in env.agents
                }
            obs, rew, term, trunc, info = env.step(actions)
            ep_ret += float(sum(rew.values()))
            for a in env.agents:
                ep_coll += float(info[a].get("collision", 0))
            if any(term.values()) or any(trunc.values()):
                successes.append(1.0 if env.all_tasks_complete else 0.0)
                break
        returns.append(ep_ret)
        collisions.append(ep_coll)

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_collisions": float(np.mean(collisions)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train PPO on logistics ParallelEnv")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-vec", type=int, default=4, help="Number of parallel vec envs")
    p.add_argument("--num-cpus", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT / "results" / "models" / "ppo_logistics"),
    )
    args = p.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
        num_agents=args.num_agents,
        seed=args.seed,
        n_vec=args.n_vec,
        num_cpus=args.num_cpus,
    )

    model = PPO(
        MlpPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(ROOT / "results" / "tensorboard"),
    )

    model.learn(total_timesteps=args.timesteps)
    zip_path = model_path.with_suffix(".zip")
    model.save(str(model_path))
    train_env.close()

    trained = PPO.load(str(model_path))

    rand_stats = evaluate_policy(
        args.num_agents, None, args.eval_episodes, seed=args.seed + 1, deterministic=True
    )
    pol_stats = evaluate_policy(
        args.num_agents,
        trained,
        args.eval_episodes,
        seed=args.seed + 1,
        deterministic=True,
    )

    bench = ROOT / "results" / "benchmarks.csv"
    bench.parent.mkdir(parents=True, exist_ok=True)
    with bench.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "setting",
                "mean_return",
                "success_rate",
                "mean_collisions",
                "timesteps",
                "num_agents",
            ]
        )
        w.writerow(
            [
                "random",
                rand_stats["mean_return"],
                rand_stats["success_rate"],
                rand_stats["mean_collisions"],
                args.timesteps,
                args.num_agents,
            ]
        )
        w.writerow(
            [
                "ppo_shared",
                pol_stats["mean_return"],
                pol_stats["success_rate"],
                pol_stats["mean_collisions"],
                args.timesteps,
                args.num_agents,
            ]
        )

    print("\n=== Evaluation (mean over episodes) ===")
    print(f"Random policy:  return={rand_stats['mean_return']:.2f}, "
          f"success={rand_stats['success_rate']:.2%}, "
          f"collisions/ep={rand_stats['mean_collisions']:.2f}")
    print(f"PPO (shared):   return={pol_stats['mean_return']:.2f}, "
          f"success={pol_stats['success_rate']:.2%}, "
          f"collisions/ep={pol_stats['mean_collisions']:.2f}")
    print(f"Model saved to {zip_path}")


if __name__ == "__main__":
    main()

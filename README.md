# Multi-Agent Logistics Task Allocator

**Multi-agent reinforcement learning (MARL) demo** for cooperative pickup–delivery in a grid-world logistics simulator. Built with [PettingZoo](https://pettingzoo.farama.org/) (multi-agent `ParallelEnv`) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (shared **PPO** policy via [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) vectorization).

It is a **prototype** for agentic task management in logistics: decentralized agents coordinate under uncertainty (traffic cells, collisions), with rewards for **efficiency** (deliveries) and **safety** (collision penalties).

---

## Demo

| | |
|---|---|
| **GIF** | Add `results/demo.gif` after training (see [Scripts](#scripts)). Placeholder: run the notebook or `scripts/render_gif.py` for a short rollout. |

### Example results (after training)

| Setting | Mean return (3 agents) | Task success rate | Collisions / episode |
|--------|-------------------------|-------------------|----------------------|
| Random policy | — | low | high |
| Shared PPO | — | higher | lower |

*Fill the table from `results/benchmarks.csv` after `agents/train_marl.py`.*

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Quick smoke test (env only, no training):

```bash
python -c "from envs.logistics_multi_env import parallel_env; e=parallel_env(num_agents=3); e.reset(seed=0); print('agents', e.agents)"
```

Train **shared PPO** on CPU (default ~100k steps; adjust `--timesteps`):

```bash
python agents/train_marl.py --timesteps 100000 --num-agents 3
```

Outputs:

- `results/models/ppo_logistics.zip` — trained policy
- `results/benchmarks.csv` — random vs PPO metrics
- `results/tensorboard/` — TensorBoard logs

```bash
tensorboard --logdir results/tensorboard
```

Interactive notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Repo layout

```
multi-agent-logistics-rl/
├── README.md
├── requirements.txt
├── envs/
│   ├── __init__.py
│   └── logistics_multi_env.py
├── agents/
│   └── train_marl.py
├── notebooks/
│   └── demo.ipynb
├── scripts/
│   └── render_gif.py
└── results/
    ├── models/
    ├── benchmarks.csv
    └── demo.gif
```

---

## Environment

- **Grid:** configurable size (default 10×10); pickup (green), delivery (blue), optional **traffic** cells (higher step cost).
- **Agents:** 3–5 (default 3); each has a fixed pickup–delivery pair; **shared** observation size (padded) for one policy.
- **Actions:** noop, move 4-way, pickup, deliver.
- **Rewards:** delivery bonus, step/traffic penalties, **collision** penalty when movement is blocked.

---

## Research / PhD angle

This project aligns with **multi-agent coordination**, **planning under uncertainty**, and **dynamic task allocation**—useful as a **small, reproducible** bridge between logistics-style applications and **agentic AI** (orchestration, RL, distributed decision-making). It is **not** a full QMIX/MAPPO implementation; the shared PPO baseline is a clear, extensible starting point.

---

## Scripts

**Generate a demo GIF** (requires a trained policy + `imageio`):

```bash
python scripts/render_gif.py --model results/models/ppo_logistics --output results/demo.gif
```

---

## Troubleshooting

- **`ConcatVecEnv` has no attribute `seed`:** Training uses manual RNG seeding (`random` / `numpy` / `torch`) instead of `PPO(seed=...)`, because SuperSuit’s stacked vec env does not implement `.seed()`.
- **`num_agents` property has no setter:** The env stores the player count as `_n_agents` to avoid clashing with PettingZoo’s read-only `num_agents`.
- **Matplotlib cache warnings:** Set `export MPLCONFIGDIR=/tmp/mpl` (or any writable folder) if the default config path is not writable.
- **TensorBoard:** Training skips TensorBoard logging if `tensorboard` is not installed; install it (`pip install tensorboard` or use `requirements.txt`) to get `results/tensorboard/` logs.

---

## License

MIT (or your choice; add a `LICENSE` file if you publish).

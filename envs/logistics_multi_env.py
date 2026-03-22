"""
PettingZoo ParallelEnv: cooperative logistics task allocation on a grid.

Agents each have a pickup → delivery task, optional traffic cells (higher step cost),
and collision penalties. Observations are padded so all agents share the same space
(for SuperSuit + Stable-Baselines3 single shared policy).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


class LogisticsMultiEnv(ParallelEnv):
    """Grid-world logistics: multiple agents pick up packages and deliver them."""

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "name": "logistics_multi_v0",
    }

    # noop, up, down, left, right, pickup, deliver
    _ACTION_NOOP = 0
    _ACTION_UP = 1
    _ACTION_DOWN = 2
    _ACTION_LEFT = 3
    _ACTION_RIGHT = 4
    _ACTION_PICKUP = 5
    _ACTION_DELIVER = 6

    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 10,
        max_steps: int = 200,
        num_traffic_cells: int = 8,
        step_penalty: float = 0.01,
        traffic_extra_penalty: float = 0.01,
        collision_penalty: float = 1.0,
        delivery_reward: float = 10.0,
        invalid_action_penalty: float = 0.1,
        render_mode: Optional[str] = None,
        max_supported_agents: int = 5,
    ):
        super().__init__()
        if not 1 <= num_agents <= max_supported_agents:
            raise ValueError("num_agents must be in [1, max_supported_agents]")
        if grid_size < 6:
            raise ValueError("grid_size should be at least 6 for valid layouts")

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_traffic_cells = num_traffic_cells
        self.step_penalty = float(step_penalty)
        self.traffic_extra_penalty = float(traffic_extra_penalty)
        self.collision_penalty = float(collision_penalty)
        self.delivery_reward = float(delivery_reward)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.render_mode = render_mode
        self.max_supported_agents = max_supported_agents

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents: List[str] = []

        self._np_random: Optional[np.random.Generator] = None
        self._step_count = 0

        # State (filled in reset)
        self._pos: Dict[str, Tuple[int, int]] = {}
        self._carrying: Dict[str, bool] = {}
        self._done_task: Dict[str, bool] = {}
        self._pickups: List[Tuple[int, int]] = []
        self._drops: List[Tuple[int, int]] = []
        self._traffic: np.ndarray = np.zeros((grid_size, grid_size), dtype=bool)
        self._grid_occ: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int8)

        self._obs_dim = self._compute_obs_dim()
        obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )
        act_space = spaces.Discrete(7)

        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

        self._surface = None  # lazy matplotlib figure for human mode

    @property
    def all_tasks_complete(self) -> bool:
        return all(self._done_task[a] for a in self.possible_agents)

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def _compute_obs_dim(self) -> int:
        # self xy, carrying, rel pickup xy, rel drop xy, task done flag,
        # padded other-agent relative xy (max_supported_agents - 1) * 2
        return 2 + 1 + 2 + 2 + 1 + 2 * (self.max_supported_agents - 1)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self._np_random = _rng(seed)
        self._step_count = 0
        self.agents = self.possible_agents[:]

        self._layout_scene()
        observations = {a: self._obs(a) for a in self.agents}
        infos = {a: {"episode_step": 0} for a in self.agents}
        return observations, infos

    def _layout_scene(self) -> None:
        rng = self._np_random
        assert rng is not None

        # Task endpoints: keep pickups and drops disjoint and spread out
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        rng.shuffle(all_cells)
        take = self.num_agents * 2
        chosen = all_cells[:take]
        self._pickups = chosen[: self.num_agents]
        self._drops = chosen[self.num_agents : self.num_agents * 2]

        used = set(self._pickups) | set(self._drops)
        free = [c for c in all_cells if c not in used]
        rng.shuffle(free)
        self._traffic[:] = False
        for i in range(min(self.num_traffic_cells, len(free))):
            x, y = free[i]
            self._traffic[x, y] = True

        spawn = [c for c in free[self.num_traffic_cells :] if c not in used]
        rng.shuffle(spawn)
        for i, a in enumerate(self.possible_agents):
            self._pos[a] = spawn[i]
            self._carrying[a] = False
            self._done_task[a] = False

        self._rebuild_occupancy()

    def _rebuild_occupancy(self) -> None:
        self._grid_occ.fill(0)
        for a, (x, y) in self._pos.items():
            self._grid_occ[x, y] += 1

    def _obs(self, agent: str) -> np.ndarray:
        g = float(self.grid_size)
        ax, ay = self._pos[agent]
        vec = np.zeros((self._obs_dim,), dtype=np.float32)
        vec[0] = ax / g
        vec[1] = ay / g
        vec[2] = 1.0 if self._carrying[agent] else 0.0

        idx = self.possible_agents.index(agent)
        if self._done_task[agent]:
            vec[3:7] = 0.0
        else:
            px, py = self._pickups[idx]
            dx, dy = self._drops[idx]
            vec[3] = (px - ax) / g
            vec[4] = (py - ay) / g
            vec[5] = (dx - ax) / g
            vec[6] = (dy - ay) / g
        vec[7] = 1.0 if self._done_task[agent] else 0.0

        # Other agents relative positions (padded)
        base = 8
        others = [x for x in self.possible_agents if x != agent]
        for j, o in enumerate(others):
            if j >= self.max_supported_agents - 1:
                break
            ox, oy = self._pos[o]
            vec[base + 2 * j] = (ox - ax) / g
            vec[base + 2 * j + 1] = (oy - ay) / g
        return vec

    def _move_delta(self, action: int) -> Tuple[int, int]:
        if action == self._ACTION_UP:
            return 0, -1
        if action == self._ACTION_DOWN:
            return 0, 1
        if action == self._ACTION_LEFT:
            return -1, 0
        if action == self._ACTION_RIGHT:
            return 1, 0
        return 0, 0

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _resolve_positions(
        self,
        old: Dict[str, Tuple[int, int]],
        proposed: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tuple[int, int]]:
        """Resolve simultaneous moves: shared destinations and blocked swaps."""
        blocked = set()
        dest_to = defaultdict(list)
        for a in self.agents:
            dest_to[proposed[a]].append(a)
        for agents in dest_to.values():
            if len(agents) > 1:
                blocked.update(agents)

        for a in self.agents:
            if a in blocked:
                continue
            dest = proposed[a]
            for b in self.agents:
                if b == a:
                    continue
                if old[b] == dest and proposed[b] == old[b]:
                    blocked.add(a)
                    break

        final: Dict[str, Tuple[int, int]] = {}
        for a in self.agents:
            final[a] = old[a] if a in blocked else proposed[a]
        return final

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self._step_count += 1
        rewards = {a: 0.0 for a in self.agents}
        collisions: Dict[str, int] = {a: 0 for a in self.agents}

        old_pos = {a: self._pos[a] for a in self.agents}
        proposed: Dict[str, Tuple[int, int]] = {}
        for a in self.agents:
            act = int(actions[a])
            x, y = self._pos[a]
            if act in (
                self._ACTION_UP,
                self._ACTION_DOWN,
                self._ACTION_LEFT,
                self._ACTION_RIGHT,
            ):
                dx, dy = self._move_delta(act)
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    proposed[a] = (nx, ny)
                else:
                    proposed[a] = (x, y)
                    rewards[a] -= self.invalid_action_penalty
            else:
                proposed[a] = (x, y)

        final_pos = self._resolve_positions(old_pos, proposed)
        for a in self.agents:
            if proposed[a] != old_pos[a] and final_pos[a] == old_pos[a]:
                rewards[a] -= self.collision_penalty
                collisions[a] = 1

        for a in self.agents:
            self._pos[a] = final_pos[a]
        self._rebuild_occupancy()

        # Pickup / deliver
        for a in self.agents:
            act = int(actions[a])
            idx = self.possible_agents.index(a)
            px, py = self._pickups[idx]
            dx, dy = self._drops[idx]

            if act == self._ACTION_PICKUP:
                if self._done_task[a]:
                    rewards[a] -= self.invalid_action_penalty
                elif self._pos[a] == (px, py) and not self._carrying[a]:
                    self._carrying[a] = True
                else:
                    rewards[a] -= self.invalid_action_penalty

            elif act == self._ACTION_DELIVER:
                if self._done_task[a]:
                    rewards[a] -= self.invalid_action_penalty
                elif self._pos[a] == (dx, dy) and self._carrying[a]:
                    self._carrying[a] = False
                    self._done_task[a] = True
                    rewards[a] += self.delivery_reward
                else:
                    rewards[a] -= self.invalid_action_penalty

            # Step / traffic cost on every timestep
            x, y = self._pos[a]
            base = self.step_penalty
            if self._traffic[x, y]:
                base += self.traffic_extra_penalty
            rewards[a] -= base

        observations = {a: self._obs(a) for a in self.agents}
        all_done = all(self._done_task[a] for a in self.possible_agents)
        timeout = self._step_count >= self.max_steps

        if all_done:
            terminations = {a: True for a in self.agents}
            truncations = {a: False for a in self.agents}
        elif timeout:
            terminations = {a: False for a in self.agents}
            truncations = {a: True for a in self.agents}
        else:
            terminations = {a: False for a in self.agents}
            truncations = {a: False for a in self.agents}

        infos = {
            a: {
                "episode_step": self._step_count,
                "collision": collisions[a],
                "delivered": int(self._done_task[a]),
            }
            for a in self.agents
        }
        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        rgb = self._render_frame()
        if self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                return rgb
            if self._surface is None:
                self._surface = plt.figure(figsize=(5, 5))
            plt.figure(self._surface.number)
            plt.clf()
            plt.imshow(rgb)
            plt.axis("off")
            plt.tight_layout()
            plt.pause(0.001)
        return rgb

    def _render_frame(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        scale = 32
        s = self.grid_size
        h, w = s * scale, s * scale
        img = Image.new("RGB", (w, h), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        for x in range(s):
            for y in range(s):
                x0, y0 = x * scale, y * scale
                x1, y1 = x0 + scale - 1, y0 + scale - 1
                fill = (255, 255, 255)
                if self._traffic[x, y]:
                    fill = (255, 243, 200)
                draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(220, 220, 220))

        colors_pickup = (80, 180, 120)
        colors_drop = (80, 120, 200)
        for (x, y) in self._pickups:
            x0, y0 = x * scale + 4, y * scale + 4
            x1, y1 = x0 + scale - 10, y0 + scale - 10
            draw.rectangle([x0, y0, x1, y1], fill=colors_pickup)
        for (x, y) in self._drops:
            x0, y0 = x * scale + 4, y * scale + 4
            x1, y1 = x0 + scale - 10, y0 + scale - 10
            draw.rectangle([x0, y0, x1, y1], fill=colors_drop)

        agent_colors = [
            (220, 60, 60),
            (60, 120, 220),
            (160, 60, 200),
            (240, 140, 40),
            (40, 180, 160),
        ]
        for i, a in enumerate(self.possible_agents):
            x, y = self._pos[a]
            cx = x * scale + scale // 2
            cy = y * scale + scale // 2
            r = scale // 3
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=agent_colors[i % len(agent_colors)],
                outline=(20, 20, 20),
            )
        return np.asarray(img)

    def close(self) -> None:
        if self._surface is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._surface)
            except Exception:
                pass
            self._surface = None


def parallel_env(**kwargs) -> LogisticsMultiEnv:
    return LogisticsMultiEnv(**kwargs)

"""
foraging_env.py -- Stage 1 (lockstep) foraging environment.

A 2D grid world. The avatar must eat food to maintain energy while avoiding
enemies that damage it on contact. Energy decays each tick (metabolic cost),
so passivity is not a viable strategy -- this is not just an obstacle course,
it's a genuine two-failure-mode survival problem (starvation vs. predation).

Critically: the avatar does NOT see the whole map. It senses only within a
limited radius, bucketed into 8 directional sectors -- partial observability
by design, so that persistent state has an actual computational job to do
(remember where food/threats were when they're no longer in view), unlike
tic-tac-toe, where full observability meant persistent memory turned out to
make no difference at all.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List

GRID_SIZE = 20
SENSE_RADIUS = 5
N_FOOD = 8
N_ENEMY = 3
ENEMY_DETECT_RADIUS = 4      # enemies start pursuing the avatar within this range
MAX_ENERGY = 100
METABOLIC_COST = 1
FOOD_ENERGY = 30
ENEMY_DAMAGE = 40
MAX_TICKS = 500

# 8 compass sectors, in fixed order: N, NE, E, SE, S, SW, W, NW
SECTOR_VECTORS = np.array([
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
], dtype=float)
SECTOR_ANGLES = np.arctan2(SECTOR_VECTORS[:, 1], SECTOR_VECTORS[:, 0])

# action indices: 0=stay, 1=N, 2=E, 3=S, 4=W
ACTIONS = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
N_ACTIONS = len(ACTIONS)


def sector_of(dx: int, dy: int) -> int:
    angle = np.arctan2(dy, dx)
    diffs = np.abs(np.angle(np.exp(1j * (angle - SECTOR_ANGLES))))
    return int(np.argmin(diffs))


class ForagingEnv:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        self.energy = MAX_ENERGY
        self.tick_count = 0
        self.food = [self._random_pos() for _ in range(N_FOOD)]
        self.enemies = [self._random_pos() for _ in range(N_ENEMY)]
        self.alive = True
        self.death_cause = None   # 'starvation' or 'predation'
        self.food_eaten = 0
        return self.observe()

    def _random_pos(self) -> np.ndarray:
        return self.rng.integers(0, GRID_SIZE, size=2)

    def observe(self) -> np.ndarray:
        """16 directional sensor neurons (8 sectors x {food, enemy} presence
        within SENSE_RADIUS) + 5 energy-level thermometer neurons = 21 total."""
        food_sectors = np.zeros(8)
        enemy_sectors = np.zeros(8)

        for f in self.food:
            d = f - self.pos
            dist = np.abs(d).max()
            if 0 < dist <= SENSE_RADIUS:
                food_sectors[sector_of(d[0], d[1])] = 1.0

        for e in self.enemies:
            d = e - self.pos
            dist = np.abs(d).max()
            if 0 < dist <= SENSE_RADIUS:
                enemy_sectors[sector_of(d[0], d[1])] = 1.0

        energy_thermometer = np.zeros(5)
        frac = self.energy / MAX_ENERGY
        n_on = int(np.ceil(frac * 5))
        energy_thermometer[:n_on] = 1.0

        return np.concatenate([food_sectors, enemy_sectors, energy_thermometer])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.alive, "step() called after death -- call reset() first"
        energy_before = self.energy

        dx, dy = ACTIONS[action]
        self.pos = np.clip(self.pos + np.array([dx, dy]), 0, GRID_SIZE - 1)

        # metabolic cost
        self.energy -= METABOLIC_COST

        # eat any food at current position, respawn it elsewhere
        for i, f in enumerate(self.food):
            if np.array_equal(f, self.pos):
                self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY)
                self.food[i] = self._random_pos()
                self.food_eaten += 1

        # move enemies: pursue if avatar within detection radius, else random walk
        for i, e in enumerate(self.enemies):
            d = self.pos - e
            dist = np.abs(d).max()
            if dist <= ENEMY_DETECT_RADIUS:
                step_dir = np.sign(d)
                step_dir = np.array([step_dir[0] if step_dir[0] != 0 else 0,
                                      step_dir[1] if step_dir[1] != 0 else 0])
                # only move one axis at a time (matches avatar's own movement rules)
                if step_dir[0] != 0 and (step_dir[1] == 0 or self.rng.random() < 0.5):
                    step_dir[1] = 0
                else:
                    step_dir[0] = 0
            else:
                choices = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
                step_dir = np.array(choices[self.rng.integers(len(choices))])
            self.enemies[i] = np.clip(e + step_dir, 0, GRID_SIZE - 1)

        # enemy contact damage
        for e in self.enemies:
            if np.array_equal(e, self.pos):
                self.energy -= ENEMY_DAMAGE

        self.tick_count += 1
        reward = self.energy - energy_before   # direct energy-change shaping

        done = False
        if self.energy <= 0:
            self.alive = False
            done = True
            self.death_cause = 'predation' if reward <= -ENEMY_DAMAGE + METABOLIC_COST else 'starvation'
        elif self.tick_count >= MAX_TICKS:
            done = True

        info = {'food_eaten': self.food_eaten, 'tick': self.tick_count,
                'death_cause': self.death_cause, 'energy': self.energy}
        return self.observe(), float(reward), done, info


N_INPUT = 21
N_OUTPUT = N_ACTIONS

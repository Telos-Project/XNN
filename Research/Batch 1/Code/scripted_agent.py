"""
scripted_agent.py -- Heuristic baseline: flee sensed enemies, otherwise move
toward the nearest sensed food, otherwise move randomly. Direct analogue of
the one-ply-lookahead opponent used for tic-tac-toe: gives an interpretable
"is this doing something sensible at all" reference point beyond beating
random, decided BEFORE seeing any trained-agent results.
"""
import numpy as np
from foraging_env import ForagingEnv, N_OUTPUT, SECTOR_VECTORS


def scripted_action(obs: np.ndarray, rng: np.random.Generator) -> int:
    food_sectors = obs[:8]
    enemy_sectors = obs[8:16]

    if enemy_sectors.sum() > 0:
        # flee: move opposite the average direction of sensed enemies
        threat_dir = SECTOR_VECTORS[enemy_sectors.astype(bool)].mean(axis=0)
        flee_dir = -threat_dir
        return _direction_to_action(flee_dir, rng)

    if food_sectors.sum() > 0:
        food_dir = SECTOR_VECTORS[food_sectors.astype(bool)].mean(axis=0)
        return _direction_to_action(food_dir, rng)

    return int(rng.integers(N_OUTPUT))


def _direction_to_action(vec: np.ndarray, rng: np.random.Generator) -> int:
    # action indices: 0=stay, 1=N(0,1), 2=E(1,0), 3=S(0,-1), 4=W(-1,0)
    if abs(vec[0]) > abs(vec[1]):
        return 2 if vec[0] > 0 else 4
    elif abs(vec[1]) > 0:
        return 1 if vec[1] > 0 else 3
    else:
        return int(rng.integers(N_OUTPUT))


def run_scripted_episode(seed: int) -> dict:
    env = ForagingEnv(seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 12345)
    done = False
    while not done:
        action = scripted_action(obs, rng)
        obs, reward, done, info = env.step(action)
    return info


if __name__ == "__main__":
    results = [run_scripted_episode(seed) for seed in range(30)]
    ticks = [r['tick'] for r in results]
    food = [r['food_eaten'] for r in results]
    causes = [r['death_cause'] for r in results]
    print(f"Scripted (flee+seek) agent over 30 episodes:")
    print(f"  mean survival ticks: {np.mean(ticks):.1f}  (max possible: 500)")
    print(f"  mean food eaten: {np.mean(food):.2f}")
    print(f"  death causes: starvation={causes.count('starvation')}  "
          f"predation={causes.count('predation')}  "
          f"none(survived to max_ticks)={causes.count(None)}")

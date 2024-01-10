# -*- coding: utf -*-

"""Custom Discrete Environments from the OpenAI Gym."""

import importlib
import logging
import sys
from io import StringIO
from itertools import product

import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

helpers = importlib.import_module("helpers")

DIRECTIONS1 = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}
DIRECTIONS2 = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}

DIRECTIONS3 = {k: DIRECTIONS2[v] for k, v in DIRECTIONS1.items()}
assert DIRECTIONS3 == {"Up": "↑", "Right": "→", "Down": "↓", "Left": "←"}


@helpers.log_func_edges
class MyFrozenLakeEnv(DiscreteEnv):
    """Custom FrozenLake Implementation w/ OpenAI Gym Env.

    Modified (somewhat) heavily from OpenAI Gym `FrozenLakeEnv`:
        - https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

    Board to be dynamic based on provided "grid". Upper left will be
    start, bottom right will be goal, and between start & goal will
    contain frozen and melted ice. Grid will be NxM grid, where N and M
    are provided at initialization.

    Each time step incurs -1 reward, and stepping into the hole incurs
    -12.5 reward. Making to to the goal incurs a +100 reward. An episode
    terminates when the agent reaches the goal.

    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, N=4, M=4, pHazard=0.15, rewards=None):
        assert 4 <= N <= 32, "N must be between 4 and 32"
        assert 4 <= M <= 32, "M must be between 4 and 32"
        assert 0.05 <= pHazard <= 0.95, "pHazard must be between 0.05 and 0.95"

        np.random.seed(N * M)

        self.REWARDS = rewards or (-1, -12.5, +100)

        self.shape = (N, M)
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)
        self.end_state_index = np.ravel_multi_index((N - 1, M - 1), self.shape)

        self.start_state_tuple = np.unravel_index(self.start_state_index, self.shape)
        self.end_state_tuple = np.unravel_index(self.end_state_index, self.shape)

        nS = np.prod(self.shape)
        nA = len(DIRECTIONS1)

        self._hazard = np.zeros(self.shape, dtype=np.bool)
        for n in range(N):
            self._hazard[n, :] = np.random.choice([True, False], M, p=[pHazard, 1 - pHazard])

        self._hazard[self.start_state_tuple] = False
        self._hazard[self.end_state_tuple] = False

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(len(P)):
            position = np.unravel_index(s, self.shape)
            P[s][DIRECTIONS1["Up"]] = self._calculate_transition_prob(position, [-1, 0])
            P[s][DIRECTIONS1["Right"]] = self._calculate_transition_prob(position, [0, 1])
            P[s][DIRECTIONS1["Down"]] = self._calculate_transition_prob(position, [1, 0])
            P[s][DIRECTIONS1["Left"]] = self._calculate_transition_prob(position, [0, -1])

        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super().__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """Prevent the agent from falling out of the grid world."""
        coord[0] = max(min(coord[0], self.shape[0] - 1), 0)
        coord[1] = max(min(coord[1], self.shape[1] - 1), 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """Determine the outcome for an action."""

        def _d(delta, d="r"):
            others = {"u": [-1, 0], "d": [1, 0], "dd": [2, 0]}
            if d in others.keys():
                return others[d]

            r = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
            l = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]

            moves = r if d == "r" else delta if d != "l" else l
            for index, move in enumerate(moves[:-1]):
                if delta == move:
                    return moves[index + 1]

        probs = list()
        for prob in [(0.80, delta), (0.10, _d(delta, "r")), (0.10, _d(delta, "l"))]:
            position = self._limit_coordinates(np.array(current) + np.array(prob[1])).astype(int)
            state = np.ravel_multi_index(tuple(position), self.shape)

            done = tuple(position) == self.end_state_tuple

            reward = self.REWARDS[1] if self._hazard[tuple(position)] else self.REWARDS[0]
            reward = self.REWARDS[2] if done else reward

            probs.append((prob[0], state, reward, done))

        return probs

    @helpers.log_func_edges
    def render(self, mode="human"):
        """Render environment as text."""
        assert mode.lower() in ("human", "ansi"), "Mode must be human or ansi"
        outfile = sys.stdout if mode == "human" else StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)

            output = " H " if self._hazard[position] else " o "
            output = " T " if position == self.end_state_tuple else output
            output = " x " if self.s == s else output

            output = output.lstrip() if position[1] == 0 else output
            output = output.rstrip() + "\n" if position[1] == self.shape[1] - 1 else output

            outfile.write(output)

        outfile.write("\n")

        return None

    @helpers.log_func_edges
    def _convert_grid_text(self):
        """Convert grid to text representation."""
        grid = list()
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)

            output = "H" if self._hazard[position] else "o"
            output = "T" if position == (self.shape[0] - 1, self.shape[1] - 1) else output
            output = "x" if self.s == s else output

            grid.append(output.strip())

        return tuple(grid)


@helpers.log_func_edges
class MyCliffWalkingEnv(DiscreteEnv):
    """Custom CliffWalking Implementation w/ OpenAI Gym Env.

    Modified (somewhat) heavily from OpenAI Gym `CliffWalkingEnv`:
        - https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
        - https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py

    Board to be dynamic based on provided "grid". Bottom left will be
    start, bottom right will be goal, and between start & goal will
    contain a cliff of some fashion. Grid will be Nx(N*4) grid, where N
    and M are provided at initialization.

    Each time step incurs -0.50 reward, and stepping off the cliff incurs
    -100 reward and a reset to the start. Making to to the goal incurs a
    +50 reward. An episode terminates when the agent reaches the goal.

    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, N=5, clevels=3, rewards=None):
        assert 3 <= N <= 12, "N must be between 3 and 12"
        assert 1 <= clevels <= N - 2, "clevels must be between 1 and N-2"

        M = N * 4
        np.random.seed(N * M)

        self.REWARDS = rewards or (-0.5, -100, +50)

        self.shape = (N, M)
        self.start_state_index = np.ravel_multi_index((N - 1, 0), self.shape)
        self.end_state_index = np.ravel_multi_index((N - 1, M - 1), self.shape)

        self.start_state_tuple = np.unravel_index(self.start_state_index, self.shape)
        self.end_state_tuple = np.unravel_index(self.end_state_index, self.shape)

        nS = np.prod(self.shape)
        nA = len(DIRECTIONS1)

        self._hazard = np.zeros(self.shape, dtype=np.bool)

        def _pick_side_index():
            upper = max((self.shape[1] - 2) // np.random.choice([2, 3, 4]), 1) + 1
            return np.random.choice(range(1, upper))

        for l in range(clevels):
            self._hazard[N - 1 - l, _pick_side_index() : -_pick_side_index()] = True

        self._hazard[self.start_state_tuple] = False
        self._hazard[self.end_state_tuple] = False

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(len(P)):
            position = np.unravel_index(s, self.shape)
            P[s][DIRECTIONS1["Up"]] = self._calculate_transition_prob(position, [-1, 0])
            P[s][DIRECTIONS1["Right"]] = self._calculate_transition_prob(position, [0, 1])
            P[s][DIRECTIONS1["Down"]] = self._calculate_transition_prob(position, [1, 0])
            P[s][DIRECTIONS1["Left"]] = self._calculate_transition_prob(position, [0, -1])

        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super().__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """Prevent the agent from falling out of the grid world."""
        coord[0] = max(min(coord[0], self.shape[0] - 1), 0)
        coord[1] = max(min(coord[1], self.shape[1] - 1), 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """Determine the outcome for an action."""

        def _d(delta, d="r"):
            others = {"u": [-1, 0], "d": [1, 0], "dd": [2, 0]}
            if d in others.keys():
                return others[d]

            r = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
            l = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]

            moves = r if d == "r" else delta if d != "l" else l
            for index, move in enumerate(moves[:-1]):
                if delta == move:
                    return moves[index + 1]

        probs = list()
        for prob in [(0.95, delta), (0.04, _d(delta, "d")), (0.01, _d(delta, "dd"))]:
            position = self._limit_coordinates(np.array(current) + np.array(prob[1])).astype(int)

            state = np.ravel_multi_index(tuple(position), self.shape)
            state = self.start_state_index if self._hazard[tuple(position)] else state

            done = tuple(position) == self.end_state_tuple

            reward = self.REWARDS[1] if self._hazard[tuple(position)] else self.REWARDS[0]
            reward = self.REWARDS[2] if done else reward

            probs.append((prob[0], state, reward, done))

        return probs

    @helpers.log_func_edges
    def render(self, mode="human"):
        """Render environment as text."""
        assert mode.lower() in ("human", "ansi"), "Mode must be human or ansi"
        outfile = sys.stdout if mode == "human" else StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)

            output = " H " if self._hazard[position] else " o "
            output = " T " if position == self.end_state_tuple else output
            output = " x " if self.s == s else output

            output = output.lstrip() if position[1] == 0 else output
            output = output.rstrip() + "\n" if position[1] == self.shape[1] - 1 else output

            outfile.write(output)

        outfile.write("\n")

        return None

    @helpers.log_func_edges
    def _convert_grid_text(self):
        """Convert grid to text representation."""
        grid = list()
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)

            output = "C" if self._hazard[position] else "o"
            output = "T" if position == (self.shape[0] - 1, self.shape[1] - 1) else output
            output = "x" if self.s == s else output

            grid.append(output.strip())

        return tuple(grid)


@helpers.log_func_edges
def transitions_and_rewards(env):
    """Build transitions & rewards matrices for environment."""
    P = np.zeros((env.nA, env.nS, env.nS))
    R = np.zeros((env.nA, env.nS, env.nS))

    for s, a in product(range(env.nS), range(env.nA)):
        for index in env.P[s][a]:
            P[a][s][index[1]] += index[0]
            R[a][s][index[1]] += index[2]

    return P, R


@helpers.log_func_edges
def render_and_policy(env, policy, mode="human"):
    """Render env and policy as text."""
    assert mode.lower() in ("human", "ansi"), "Mode must be human or ansi"
    outfile = sys.stdout if mode == "human" else StringIO()

    assert env.nS == len(policy), "Length of policy must equal env.nS"

    print(f"Rendering grid and policy for {type(env).__name__}")

    base = np.array(env._convert_grid_text())
    policy = np.array(tuple([DIRECTIONS2[p] for p in policy]))

    for index, char in enumerate(base):
        if char == "C":
            policy[index] = char

    base = base.reshape(env.shape)
    policy = policy.reshape(env.shape)

    for index in range(env.shape[0]):
        outfile.write(f"{' '.join(base[index])}  -->  {' '.join(policy[index])}\n")

    outfile.write("\n")

    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s:L%(lineno)s - %(message)s",
    )

    logging.info("----- FrozenLake Env :: Small Env Problem -----")
    envSmall = MyFrozenLakeEnv()
    envSmall.render()

    logging.info("----- CliffWalking Env :: Large Env Problem -----")
    envLarge = MyCliffWalkingEnv()
    envLarge.render()

import collections
import numpy as np
import torch

device = torch.device("cpu")


# Adapted from OpenAI: https://github.com/openai/
class ReplayBuffer(object):
    def __init__(self, size):
        self._memory = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._memory)

    def add(self, obs_t, action, reward, obs_tp1, done):
        if self._next_idx >= len(self._memory):
            self._memory.append((obs_t, action, reward, obs_tp1, done))
        else:
            self._memory[self._next_idx] = (obs_t, action, reward, obs_tp1, done)
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            data = self._memory[i]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        s = torch.stack(states).float().to(device)
        a = torch.tensor(actions, dtype=torch.long, device=device)
        r = torch.tensor(rewards, dtype=torch.float, device=device)
        s2 = torch.stack(next_states).float().to(device)
        d = torch.tensor(dones, dtype=torch.bool, device=device)
        return s, a, r, s2, d

    def sample(self, batch_size):
        idxes = list(np.random.randint(len(self._memory), size=batch_size))
        return self._encode_sample(idxes)


class LinearMemory:
    def __init__(self, buffer_size: int, batch_size: int):
        # self.buffer_size = buffer_size
        # self.batch_size = batch_size
        field_names = ["state", "action", "reward", "next_state", "done"]
        self.ExperienceType = collections.namedtuple("Experience", field_names=field_names)
        self.memory = collections.deque()

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.ExperienceType(state, action, reward, next_state, done))

    def sample(self, sample_all: bool = False, as_torch: bool = True):
        experiences = list(self.memory)
        self.memory.clear()

        # TODO - if not broken - remove this
        # Clear empty experiences
        # experiences = [e for e in experiences if e is not None]
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones = [e.done for e in experiences]
        if not as_torch:
            return states, actions, rewards, next_states, dones
        else:
            states = torch.from_numpy(np.vstack(states)).float().to(device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(device)
            rewards = torch.tensor(rewards).float().to(device)
            next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
            dones = torch.tensor(dones).bool().to(device)
            return states, actions, rewards, next_states, dones

import numpy as np
import torch


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class Runner(object):
    """
    Copied from OpenAI Baseline
    """

    def __init__(self, env, model, n_step, gamma=0.99):
        self.env = env
        n_env = self.env.num_envs
        self.model = model
        self.n_step = n_step
        self.gamma = gamma

        obs = env.reset()
        self.obs = obs.transpose(0, 3, 1, 2)
        # This state is the LSTM hidden state, not used for non-lstm situation
        self.lstm_hidden_state = model.lstm_initial_state
        self.dones = np.array([False for _ in range(n_env)])

        self.batch_ob_shape = (n_env * n_step, 4, 84, 84)

    def run(self):
        memory_obs, memory_rewards, memory_actions, memory_dones = [], [], [], []
        memory_lstm_state = self.lstm_hidden_state

        for n in range(self.n_step):
            actions, values, lstm_hidden_state = self.model.forward(
                self.obs,
                lstm_states=self.lstm_hidden_state,
                masks=self.dones,
                training=False)  # Both actor and critic
            memory_obs.append(np.copy(self.obs))
            memory_actions.append(actions)
            memory_dones.append(self.dones)
            self.lstm_hidden_state = lstm_hidden_state

            new_obs, rewards, dones, _ = self.env.step(np.array(actions))
            new_obs = new_obs.transpose(0, 3, 1, 2)
            self.dones = dones
            self.obs = new_obs
            memory_rewards.append(rewards)

        memory_dones.append(self.dones)
        memory_obs = np.asarray(
            memory_obs, dtype=np.uint8).swapaxes(1,
                                                 0).reshape(self.batch_ob_shape)
        memory_rewards = np.asarray(
            memory_rewards, dtype=np.float64).swapaxes(1, 0)
        memory_dones = np.asarray(memory_dones, dtype=np.bool).swapaxes(1, 0)
        # Only memory masks is not flattened and transposed here, this is inconsistent...
        memory_masks = memory_dones[:, :-1]
        memory_dones = memory_dones[:, 1:]

        # Discount / bootstrap from last value function
        _, last_values, _ = self.model.forward(
            self.obs,
            lstm_states=self.lstm_hidden_state,
            masks=self.dones,
            training=False)
        last_values = list(last_values.detach())
        for n, (rewards, dones, value) in enumerate(
                zip(memory_rewards, memory_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            memory_rewards[n] = rewards

        memory_discounted_rewards = memory_rewards.flatten()
        tensor_actions = torch.stack(memory_actions).transpose(1, 0)
        tensor_actions = tensor_actions.contiguous().view(-1)
        tensor_discounted_reward = torch.Tensor(memory_discounted_rewards).to(
            tensor_actions.device)

        return memory_obs, tensor_discounted_reward, tensor_actions, memory_lstm_state, memory_masks

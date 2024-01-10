# Thin wrapper around cartpole from OpenAI's Gym toolkit
# This env models a cart with a pole balancing on top of it
import agentos
import gym
import numpy as np
from dm_env import specs
from dm_env import TimeStep
from dm_env import StepType


class CartPole(agentos.Environment):
    def __init__(self, **kwargs):
        self.discount = np.float32(kwargs["discount"])
        self.cartpole = gym.make("CartPole-v1")
        self.reset()

    def step(self, action):
        assert action in [0, 1]
        result = self.cartpole.step(action)
        self.last_obs, self.last_reward, self.done, self.info = result
        # FIXME - this cast makes it match spec
        # https://github.com/deepmind/acme/blob/061c400aaa038d4dcfa34cb7438f4fcbeca52ac2/acme/wrappers/gym_wrapper.py#L60
        observation = np.float32(self.last_obs)
        reward = np.float32(self.last_reward)
        # https://github.com/deepmind/dm_env/blob/738783150dfc74ac0b878fe76fb7f73caf3b3898/dm_env/_environment.py#L228
        if self.done:
            if self.info.get("TimeLimit.truncated", False):
                return TimeStep(
                    StepType.LAST, reward, self.discount, observation
                )
            return TimeStep(
                StepType.LAST, reward, np.float32(0.0), observation
            )
        return TimeStep(StepType.MID, reward, self.discount, observation)

    @property
    def valid_actions(self):
        return [0, 1]

    def reset(self):
        self.last_obs = None
        self.last_reward = None
        self.last_done = False
        self.last_info = None
        self.last_obs = self.cartpole.reset()
        # FIXME - this cast makes it match spec
        return TimeStep(StepType.FIRST, None, None, np.float32(self.last_obs))

    def get_spec(self):
        observations = specs.BoundedArray(
            shape=(4,),
            dtype=np.dtype("float32"),
            name="observation",
            minimum=[
                -4.8000002e00,
                -3.4028235e38,
                -4.1887903e-01,
                -3.4028235e38,
            ],
            maximum=[4.8000002e00, 3.4028235e38, 4.1887903e-01, 3.4028235e38],
        )
        actions = specs.DiscreteArray(num_values=2)
        discounts = specs.BoundedArray(
            shape=(),
            dtype=np.dtype("float32"),
            name="discount",
            minimum=0.0,
            maximum=1.0,
        )
        return agentos.EnvironmentSpec(
            observations=observations,
            actions=actions,
            rewards=self.reward_spec(),
            discounts=discounts,
        )

    def reward_spec(self):
        return specs.Array(shape=(), dtype=np.dtype("float32"), name="reward")


# Unit test for Cartpole
def run_tests():
    print("Testing Cartpole...")
    env = CartPole(discount=0.99)
    spec = env.get_spec()
    assert spec is not None
    assert spec.observations is not None
    assert spec.actions is not None
    assert spec.rewards is not None
    assert spec.discounts is not None
    first_obs = env.reset()
    assert len(first_obs) == 4
    obs, reward, done, info = env.step(0)
    obs, reward, done, info = env.step(1)
    while not done:
        obs, reward, done, info = env.step(1)
    print("Test successful...")


if __name__ == "__main__":
    run_tests()

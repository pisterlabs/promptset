from typing import Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from distrax import MultivariateNormalDiag

from pax import utils


class ContinuousValueHead(hk.Module):
    """Network head that produces a continuous distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self._hidden_actor = hk.Linear(256)
        self._policy_layer = hk.Linear(
            2 * num_values,
            w_init=hk.initializers.Orthogonal(0.01),  # baseline
            b_init=hk.initializers.Constant(0.0),
            name="critic",
        )

        self._hidden_critic = hk.Linear(256)

        self._critic_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),  # baseline
            with_bias=False,
        )

        self.num_outputs = num_values

    def __call__(self, inputs: jnp.ndarray):

        h1 = jax.nn.tanh(self._hidden_actor(inputs))
        pi = self._policy_layer(h1)
        loc, scale = jnp.split(pi, [self.num_outputs], axis=-1)
        scale = jax.nn.softplus(scale)

        h1 = jax.nn.tanh(self._hidden_critic(inputs))
        value = jnp.squeeze(self._critic_layer(h1), axis=-1)
        return (
            distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale),
            value,
        )


def make_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                ContinuousValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_GRU(num_actions: int):
    hidden_size = 25
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[MultivariateNormalDiag, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        gru = hk.GRU(hidden_size)
        embedding, state = gru(inputs, state)
        logits, values = ContinuousValueHead(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def make_GRU_hypernetwork(num_actions: int):
    hidden_size = 25
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[MultivariateNormalDiag, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        gru = hk.GRU(hidden_size)
        embedding, state = gru(inputs, state)
        logits, values = ContinuousValueHead(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def test_GRU_hyper():
    key = jax.random.PRNGKey(seed=0)
    num_actions = 5
    obs_spec = (10,)
    key, subkey = jax.random.split(key)
    dummy_obs = jnp.zeros(shape=obs_spec)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    network, hidden = make_GRU_hypernetwork(num_actions)
    print(hidden.shape)
    initial_params = network.init(subkey, dummy_obs, hidden)
    print("GRU w_i", initial_params["gru"]["w_i"].shape)
    print("GRU w_h", initial_params["gru"]["w_h"].shape)
    print(
        "Policy head",
        initial_params["continuous_value_head/~/linear"]["w"].shape,
    )
    print(
        "Value head",
        initial_params["continuous_value_head/~/linear_1"]["w"].shape,
    )
    observation = jnp.zeros(shape=(1, 10))
    observation = jnp.zeros(shape=(10, 10))
    (logits, values), hidden = network.apply(
        initial_params, observation, hidden
    )
    print(hidden.shape)
    return network


if __name__ == "__main__":
    test_GRU_hyper()

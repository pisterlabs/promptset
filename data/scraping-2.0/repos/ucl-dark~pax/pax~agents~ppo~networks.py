from typing import Optional, Tuple, Any

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
from distrax import MultivariateNormalDiag, Categorical
from jax import Array

from pax import utils
from pax.utils import float_precision


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return distrax.Categorical(logits=logits), value


class CategoricalValueHead_ipd(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


class CategoricalValueHeadSeparate(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._action_body = hk.nets.MLP(
            [64, 64],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=jnp.tanh,
        )
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
        )
        self._value_body = hk.nets.MLP(
            [64, 64],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=jnp.tanh,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1),
            b_init=hk.initializers.Constant(0),
        )

    def __call__(self, inputs: jnp.ndarray):
        # action_output, value_output = inputs
        logits = self._action_body(inputs)
        logits = self._logit_layer(logits)

        value = self._value_body(inputs)
        value = jnp.squeeze(self._value_layer(value), axis=-1)
        return (distrax.Categorical(logits=logits), value)


class CategoricalValueHeadSeparate_ipditm(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        hidden_size: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._action_body = hk.nets.MLP(
            [hidden_size],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=jnp.tanh,
        )
        self._value_body = hk.nets.MLP(
            [hidden_size],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
            activation=jnp.tanh,
        )
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0),
        )

    def __call__(self, inputs: jnp.ndarray):
        # action_output, value_output = inputs
        logits = self._action_body(inputs)
        logits = self._logit_layer(logits)

        value = self._value_body(inputs)
        value = jnp.squeeze(self._value_layer(value), axis=-1)
        return distrax.Categorical(logits=logits), value


class ContinuousValueHead(hk.Module):
    """Network head that produces a continuous distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
        mean_activation: Optional[str] = None,
        with_bias=False,
    ):
        super().__init__(name=name)
        self.mean_action = mean_activation
        self._mean_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=with_bias,
        )
        self._scale_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=with_bias,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            with_bias=with_bias,
        )

    def __call__(self, inputs: jnp.ndarray):
        if self.mean_action == "sigmoid":
            means = jax.nn.sigmoid(self._mean_layer(inputs))
        else:
            means = self._mean_layer(inputs)
        scales = self._scale_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        scales = jnp.maximum(scales, 0.01)
        return (
            distrax.MultivariateNormalDiag(loc=means, scale_diag=scales),
            value,
        )


class Tabular(hk.Module):
    def __init__(self, num_values: int):
        super().__init__(name="Tabular")
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )

        def _input_to_onehot(input: jnp.ndarray):
            chunks = jnp.array([9**3, 9**2, 9, 1], dtype=jnp.int32)
            idx = input.nonzero(size=4)[0]
            idx = jnp.mod(idx, 9)
            idx = chunks * idx
            idx = jnp.sum(idx)
            return jax.nn.one_hot(idx, num_classes=6561)

        self.input_to_onehot = jax.vmap(_input_to_onehot)

    def __call__(self, inputs: jnp.ndarray):
        inputs = self.input_to_onehot(inputs)
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)

        return (distrax.Categorical(logits=logits), value)


class CNN(hk.Module):
    def __init__(self, output_channels, kernel_shape):
        super().__init__(name="CNN")
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
        )
        self.conv_a_1 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
        )
        self.linear_a_0 = hk.Linear(output_channels)

        self.flatten = hk.Flatten()

    def __call__(self, inputs: jnp.ndarray):
        # Actor and Critic
        x = self.conv_a_0(inputs)
        x = jax.nn.relu(x)
        x = self.conv_a_1(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear_a_0(x)
        x = jax.nn.relu(x)
        return x


class CNN_ipditm(hk.Module):
    def __init__(self, output_channels, kernel_shape):
        super().__init__(name="CNN")
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
        )
        # akbir suggested fix
        self.flatten = hk.Flatten()

    def __call__(self, inputs: jnp.ndarray):
        obs = inputs["observation"]
        inventory = inputs["inventory"]
        # Actor and Critic
        x = self.conv_a_0(obs)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = jnp.concatenate([x, inventory], axis=-1)
        return x


class CNNSeparate_ipditm(hk.Module):
    def __init__(self, output_channels, kernel_shape, num_actions: int):
        super().__init__(name="CNN")
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_a_0 = hk.Linear(output_channels)
        self.conv_v_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_v_0 = hk.Linear(1)
        self.flatten = hk.Flatten()

    def __call__(self, inputs):
        obs = inputs["observation"]
        inventory = inputs["inventory"]
        # Actor
        x = self.conv_a_0(obs)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = jnp.concatenate([x, inventory], axis=-1)
        logits = self.linear_a_0(x)

        # Critic
        x = self.conv_v_0(obs)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = jnp.concatenate([x, inventory], axis=-1)
        x = self.linear_v_0(x)
        val = x
        return (distrax.Categorical(logits=logits), jnp.squeeze(val, axis=-1))


def make_ipd_network(num_actions: int, tabular: bool, hidden_size: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        if tabular:
            layers.extend(
                [
                    CategoricalValueHead_ipd(num_values=num_actions),
                ]
            )
        else:
            layers.extend(
                [
                    hk.nets.MLP(
                        [hidden_size, hidden_size],
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0),
                        activate_final=True,
                        activation=jnp.tanh,
                    ),
                    CategoricalValueHead_ipd(num_values=num_actions),
                ]
            )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_cournot_network(num_actions: int, hidden_size: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                hk.nets.MLP(
                    [hidden_size, hidden_size],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                    activation=jnp.tanh,
                ),
                ContinuousValueHead(
                    num_values=num_actions, name="cournot_value_head"
                ),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_fishery_network(num_actions: int, hidden_size: int):
    """Continuous action space network with values clipped between 0 and 1"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                hk.nets.MLP(
                    [hidden_size, hidden_size],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                    activation=jnp.tanh,
                ),
                ContinuousValueHead(
                    num_values=num_actions, name="fishery_value_head"
                ),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_rice_sarl_network(num_actions: int, hidden_size: int):
    """Continuous action space network with values clipped between 0 and 1"""
    if float_precision == jnp.float16:
        policy = jmp.get_policy(
            "params=float16,compute=float16,output=float32"
        )
        hk.mixed_precision.set_policy(hk.nets.MLP, policy)

    def forward_fn(inputs):
        layers = [
            hk.nets.MLP(
                [hidden_size, hidden_size],
                w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                b_init=hk.initializers.Constant(0),
                activate_final=True,
                activation=jax.nn.relu,
            ),
            ContinuousValueHead(
                num_values=num_actions,
                name="rice_value_head",
                mean_activation="sigmoid",
            ),
        ]

        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_coingame_network(
    num_actions: int,
    tabular: bool,
    with_cnn: bool,
    separate: bool,
    hidden_size: int,
    output_channels: int,
    kernel_shape: int,
):
    def forward_fn(inputs):
        layers = []
        if tabular:
            layers.extend(
                [
                    Tabular(num_values=num_actions),
                ]
            )
        elif with_cnn:
            if separate:
                cnn = CNNSeparate_ipditm(
                    output_channels, kernel_shape, num_actions
                )
                cvh = CategoricalValueHeadSeparate(num_values=num_actions)
            else:
                cnn = CNN(output_channels, kernel_shape)
                cvh = CategoricalValueHead(num_values=num_actions)
            layers.extend([cnn, cvh])
        else:
            layers.extend(
                [
                    hk.nets.MLP(
                        [hidden_size],
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0),
                        activate_final=True,
                        activation=jnp.tanh,
                    ),
                    CategoricalValueHead(num_values=num_actions),
                ]
            )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_cartpole_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                hk.nets.MLP(
                    [64, 64],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                ),
                CategoricalValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_sarl_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend([CategoricalValueHeadSeparate(num_values=num_actions)])
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_ipditm_network(
    num_actions: int,
    separate: bool,
    with_cnn: bool,
    hidden_size: int,
    output_channels: int,
    kernel_shape: int,
):
    def forward_fn(inputs: dict):
        layers = []

        if separate and with_cnn:
            cnn = CNNSeparate_ipditm(
                output_channels, kernel_shape, num_actions
            )
            layers.extend([cnn])
        elif with_cnn:
            cnn = CNN_ipditm(output_channels, kernel_shape)
            cvh = CategoricalValueHead(num_values=num_actions)
            layers.extend([cnn, cvh])

        else:
            raise ValueError("IPDITM network must have a CNN")
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_GRU_ipd_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[Categorical, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        gru = hk.GRU(hidden_size)
        embedding, state = gru(inputs, state)
        logits, values = CategoricalValueHead_ipd(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def make_GRU_cartpole_network(num_actions: int):
    hidden_size = 256
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[Categorical, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        torso = hk.nets.MLP(
            [hidden_size, hidden_size],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
        )
        gru = hk.GRU(hidden_size)
        embedding = torso(inputs)
        embedding, state = gru(embedding, state)
        logits, values = CategoricalValueHead(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def make_GRU_coingame_network(
    num_actions: int,
    with_cnn: bool,
    hidden_size: int,
    output_channels: int,
    kernel_shape: Tuple[int],
):
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[Categorical, jnp.ndarray], jnp.ndarray]:

        if with_cnn:
            torso = CNN(output_channels, kernel_shape)(inputs)

        else:
            torso = hk.nets.MLP(
                [hidden_size],
                w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                b_init=hk.initializers.Constant(0),
                activate_final=True,
            )
        gru = hk.GRU(
            hidden_size,
            w_h_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            w_i_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
        )

        embedding = torso(inputs)
        embedding, state = gru(embedding, state)
        logits, values = CategoricalValueHead(num_actions)(embedding)

        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_GRU_ipditm_network(
    num_actions: int,
    hidden_size: int,
    separate: bool,
    output_channels: int,
    kernel_shape: Tuple[int],
):
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[Categorical, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        torso = CNN_ipditm(output_channels, kernel_shape)
        gru = hk.GRU(
            hidden_size,
            w_i_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            w_h_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            b_init=hk.initializers.Constant(0),
        )
        if separate:
            cvh = CategoricalValueHeadSeparate_ipditm(
                num_values=num_actions, hidden_size=hidden_size
            )
        else:
            cvh = CategoricalValueHead(num_values=num_actions)
        embedding = torso(inputs)
        embedding, state = gru(embedding, state)
        logits, values = cvh(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_GRU_fishery_network(
    num_actions: int,
    hidden_size: int,
):
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> tuple[tuple[MultivariateNormalDiag, Array], Any]:
        """forward function"""
        gru = hk.GRU(
            hidden_size,
            w_i_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            w_h_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            b_init=hk.initializers.Constant(0),
        )

        cvh = ContinuousValueHead(num_values=num_actions)
        embedding, state = gru(inputs, state)
        logits, values = cvh(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_GRU_rice_network(
    num_actions: int,
    hidden_size: int,
    v2=False,
):
    # if float_precision == jnp.float16:
    #     policy = jmp.get_policy('params=float16,compute=float16,output=float32')
    #     hk.mixed_precision.set_policy(hk.GRU, policy)
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> tuple[tuple[MultivariateNormalDiag, Array], Any]:
        gru = hk.GRU(
            hidden_size,
            w_i_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            w_h_init=hk.initializers.Orthogonal(jnp.sqrt(1)),
            b_init=hk.initializers.Constant(0),
        )

        if v2:
            cvh = ContinuousValueHead(
                num_values=num_actions,
                mean_activation="sigmoid",
                with_bias=True,
            )
        else:
            cvh = ContinuousValueHead(num_values=num_actions)
        embedding, state = gru(inputs, state)
        logits, values = cvh(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def test_GRU():
    key = jax.random.PRNGKey(seed=0)
    num_actions = 2
    obs_spec = (5,)
    key, subkey = jax.random.split(key)
    dummy_obs = jnp.zeros(shape=obs_spec)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    network, hidden = make_GRU_ipd_network(num_actions)
    print(hidden.shape)
    initial_params = network.init(subkey, dummy_obs, hidden)
    print("GRU w_i", initial_params["gru"]["w_i"].shape)
    print("GRU w_h", initial_params["gru"]["w_h"].shape)
    print(
        "Policy head",
        initial_params["categorical_value_head/~/linear"]["w"].shape,
    )
    print(
        "Value head",
        initial_params["categorical_value_head/~/linear_1"]["w"].shape,
    )
    observation = jnp.zeros(shape=(1, 5))
    observation = jnp.zeros(shape=(10, 5))
    (logits, values), hidden = network.apply(
        initial_params, observation, hidden
    )
    print(hidden.shape)
    return network


if __name__ == "__main__":
    test_GRU()

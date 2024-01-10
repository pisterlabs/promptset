import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, Relu, Dense, parallel, FanOut
from jax.nn.initializers import he_normal, zeros, ones

from jax import grad, jit, value_and_grad
from functools import partial

import numpy as np

import stax_nn_utils as stu
from replay_buffer import ReplayBuffer, SARnSn_from_SAR
from noise_procs import dampedSpringNoise
import jax_nn_utils as jnn

import rl_types as RT
import jax_trajectory_utils_2 as JTU
import typing as TT

from collections import namedtuple


class SAC_fns(TT.NamedTuple):
    pi: RT.NNParamsFn
    q: RT.NNParamsFn


class SAC_params(TT.NamedTuple):
    pi: RT.NNParams
    q: RT.NNParams
    q_targ: RT.NNParams
    alpha: RT.Tensor
    gamma: RT.Tensor
    H_target: RT.Tensor


class SAC_obs(TT.NamedTuple):
    S: RT.Tensor
    A: RT.Tensor
    R: RT.Tensor
    Sn: RT.Tensor
    eps: RT.Tensor


LossFn = TT.Callable[[RT.NNParams, SAC_obs, SAC_params, SAC_fns], RT.Tensor]


def create_pi_net(
    obs_dim: int, action_dim: int, rngkey=jax.random.PRNGKey(0)
) -> TT.Tuple[RT.NNParams, RT.NNParamsFn]:
    pi_init, pi_fn = serial(
        Dense(64, he_normal(), zeros),
        Relu,
        FanOut(2),
        parallel(
            serial(
                Dense(64, he_normal(), zeros),
                Relu,
                Dense(action_dim, he_normal(), zeros),
            ),
            serial(
                Dense(64, he_normal(), zeros),
                Relu,
                Dense(action_dim, he_normal(), zeros),
            ),
        ),
    )
    output_shape, pi_params = pi_init(rngkey, (1, obs_dim))
    pi_fn = jit(pi_fn)
    return pi_params, pi_fn


def create_q_net(
    obs_dim, action_dim, rngkey=jax.random.PRNGKey(0)
) -> TT.Tuple[RT.NNParams, RT.NNParamsFn]:
    q_init, q_fn = serial(
        Dense(64, he_normal(), zeros),
        Relu,
        Dense(64, he_normal(), zeros),
        Relu,
        Dense(action_dim, he_normal(), zeros),
    )
    output_shape, q_params = q_init(rngkey, (1, obs_dim + action_dim))

    @jit
    def q_fn2(q, S, A):
        return q_fn(q, jnp.hstack([S, A]))

    return q_params, q_fn2


@jit
def log_pdf_std_norm(x, mu, std):
    # only applicable when covariance is a diagonal matrix! (represented by "std" vector)
    # textbook log of normal PDF
    d = len(x)
    return -0.5 * (
        d * jnp.log(2 * jnp.pi) + jnp.sum(2 * jnp.log(std) + ((x - mu) / std) ** 2)
    )


@jit
def log_prob_tanh_std_norm(action, mu, std):
    # https://arxiv.org/pdf/1812.05905.pdf eq 26,
    # with tip from openai sac implementation
    return log_pdf_std_norm(action, mu, std) - jnp.sum(
        jnp.log(2) - action - jax.nn.softplus(-2 * action)
    )


@partial(jit, static_argnums=(3,))
def action_with_log_prob(pi, S, eps, pi_fn):
    mu, log_std = pi_fn(pi, S)
    std = jnp.exp(log_std)
    pre_action = mu + std * eps  # equivalent to u~mu(u|s), eq 25 and p 16
    log_prob = log_prob_tanh_std_norm(pre_action, mu, std)
    action = jnp.tanh(pre_action)
    return action, log_prob


@partial(jit, static_argnums=(3,))
def action(pi, S, eps, pi_fn):
    mu, log_std = pi_fn(pi, S)
    std = jnp.exp(log_std)
    return jnp.tanh(mu + std * eps)


@partial(jit, static_argnums=(3,))
def pi_loss(
    pi: RT.NNParams, obs: SAC_obs, non_grad: SAC_params, fns: SAC_fns
) -> RT.Tensor:
    S, eps = obs.S, obs.eps
    # https://arxiv.org/pdf/1812.05905.pdf eq. 7
    # TODO: add second Q function and take min! (described on p.8)
    action, log_prob = action_with_log_prob(pi, S, eps, fns.pi)
    return non_grad.alpha * log_prob - fns.q(non_grad.q, S, action)


@partial(jit, static_argnums=(3,))
def q_loss(
    q: RT.NNParams, obs: SAC_obs, non_grad: SAC_params, fns: SAC_fns
) -> RT.Tensor:
    q_fn, pi_fn = fns.q, fns.pi
    q_targ = non_grad.q_targ
    pi = non_grad.pi
    gamma = non_grad.gamma
    alpha = non_grad.alpha

    S, S_n, R, A, eps = obs.S, obs.Sn, obs.R, obs.A, obs.eps
    # implements https://arxiv.org/pdf/1812.05905.pdf eq. 5 (substituting in 3)
    # TODO: add second Q function and take min! (described on p.8)
    current_val_est = q_fn(q, S, A)

    A_n, log_prob_A_n = action_with_log_prob(pi, S_n, eps, pi_fn)

    V_n = q_fn(q_targ, S_n, A_n) - alpha * log_prob_A_n  # eq 3

    disc_future_val_est = gamma ** len(R) * V_n
    disc_rewards = jnp.sum(R * gamma ** jnp.arange(0, len(R)))
    return (current_val_est - (disc_rewards + disc_future_val_est)) ** 2


@partial(jit, static_argnums=(3,))
def alpha_loss(
    alpha: RT.NNParams, obs: SAC_obs, non_grad: SAC_params, fns: SAC_fns
) -> RT.Tensor:
    S, eps = obs.S, obs.eps
    # https://arxiv.org/pdf/1812.05905.pdf eq. 18
    action, log_prob = action_with_log_prob(non_grad.pi, S, eps, fns.pi)
    return -alpha * (log_prob + non_grad.H_target)


@partial(jit, static_argnums=(1,))
def batch_grad(params, loss_fn, batch):
    # loss_fn must be a (partially evaled) function that takes only (params, obs)
    @jit
    def g(params, batch):
        return jnp.mean(jax.vmap(loss_fn, (None, 0))(params, batch))

    return value_and_grad(g)(params, batch)


@partial(jit, static_argnums=(0, 4))
def batch_mean_loss(
    loss_fn: LossFn,
    params: RT.NNParams,
    batch: SAC_obs,
    non_grad: SAC_params,
    fns: SAC_fns,
):
    return jnp.mean(
        jax.vmap(loss_fn, (None, 0, None, None))(params, batch, non_grad, fns)
    )


@partial(jit, static_argnums=(0, 4))
def batch_grad_2(
    loss_fn: LossFn,
    params: RT.NNParams,
    batch: SAC_obs,
    non_grad: SAC_params,
    fns: SAC_fns,
) -> TT.Tuple[RT.Tensor, RT.Tensor]:
    # loss_fn must be a (partially evaled) function that takes only (params, obs)
    def batch_mean_loss(params, batch, non_grad, fns):
        return jnp.mean(
            jax.vmap(loss_fn, (None, 0, None, None))(params, batch, non_grad, fns)
        )

    return value_and_grad(batch_mean_loss)(params, batch, non_grad, fns)


class Agent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_max=2,
        memory_size=1e6,
        batch_size=256,
        td_steps=1,
        discount=0.99,
        LR=3 * 1e-4,
        tau=0.005,
        update_interval=1,
        grad_steps_per_update=1,
        seed=0,
        alpha_0=0.2,
        reward_standardizer=jnn.RewardStandardizer(),
        state_transformer=None,
    ):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_max = action_max
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.td_steps = td_steps
        self.gamma = discount
        self.LR = LR
        self.tau = tau
        self.update_interval = update_interval
        self.grad_steps_per_update = grad_steps_per_update
        self.seed = seed

        self.rngkey = jax.random.PRNGKey(seed)

        q_params, q_fn = create_q_net(obs_dim, act_dim, self.new_key())
        self.q_fn = q_fn
        self.q = q_params
        self.q_targ = stu.copy_network(q_params)

        pi_params, pi_fn = create_pi_net(obs_dim, act_dim, self.new_key())
        self.pi_fn = pi_fn
        self.pi = pi_params

        self.H_target = -act_dim
        self.memory_train = ReplayBuffer(
            obs_dim, act_dim, memory_size, reward_steps=td_steps, batch_size=batch_size
        )
        self.reward_standardizer = reward_standardizer
        self.state_transformer = state_transformer
        self.state_transformer_batched = jax.jit(jax.vmap(state_transformer))

        self.alpha = alpha_0

    def normalize_action(self, A):
        return A / self.action_max

    def new_key(self):
        self.rngkey, rngkey = jax.random.split(self.rngkey)
        return rngkey

    def new_eps(self, shape):
        return jax.random.normal(self.new_key(), shape=shape)

    def predict_pi(self, state, eps):
        pi_fn = self.pi_fn
        pi = self.pi
        state_transformer_batched = self.state_transformer_batched
        action_max = self.action_max
        return action_max * action(pi, state_transformer_batched(state), eps, pi_fn)

    def predict_pi_opt(self, S):
        return self.predict_pi(S, 0)

    def predict_q(self, S, A):
        return self.q_fn(
            self.q, self.state_transformer_batched(S), self.normalize_action(A)
        )

    def predict_action_and_log_prob(self, S, eps):
        return action_with_log_prob(
            self.pi, self.state_transformer_batched(S), eps, self.pi_fn
        )

    def make_agent_act_fn(self):
        pi_fn = self.pi_fn
        pi = self.pi
        state_transformer = self.state_transformer
        action_max = self.action_max

        if state_transformer is not None:

            def act_fn(state, eps):
                state = state_transformer(state)
                return action_max * action(pi, state, eps, pi_fn)

        else:

            def act_fn(state, eps):
                return action_max * action(pi, state, eps, pi_fn)

        return act_fn

    def make_policynet_act_fn(self) -> JTU.PolicyNetFn:
        pi_fn = self.pi_fn
        state_transformer = self.state_transformer
        action_max = self.action_max

        if state_transformer is not None:

            def act_fn(policy_net, state, eps):
                state = state_transformer(state)
                return action_max * action(policy_net, state, eps, pi_fn)

        else:

            def act_fn(policy_net, state, eps):
                return action_max * action(policy_net, state, eps, pi_fn)

        return act_fn

    def remember_episode(self, S, A, R):
        self.reward_standardizer.observe_reward_vec(R)
        R = self.reward_standardizer.standardize_reward(R)
        S, A, R1n, Sn = SARnSn_from_SAR(S, A, R, self.td_steps)
        if self.state_transformer_batched is not None:
            S = self.state_transformer_batched(S)
            Sn = self.state_transformer_batched(Sn)
        A = self.normalize_action(A)

        self.memory_train.store_many(S, A, R1n, Sn, np.zeros((S.shape[0], 1)))

    # @jit
    # def init_q_batch_grad():
    #     def q_batch_grad():
    #         q_loss_val, q_grad = batch_grad(self.q, q_loss_agent, (S, S_n, R, A, eps))

    def update(self, LR=None):
        if LR is None:
            LR = self.LR

        S, S_n, A, R, _ = self.memory_train.sample_batch()
        eps = self.new_eps(A.shape)

        @jit
        def q_loss_agent(params, obs):
            fns = (self.q_fn, self.pi_fn)
            non_grad = (self.q_targ, self.pi, self.gamma, self.alpha)
            return q_loss(params, obs, non_grad, fns)

        @jit
        def pi_loss_agent(params, obs):
            non_grad = (self.q, self.alpha)
            fns = (self.pi_fn, self.q_fn)
            return pi_loss(params, obs, non_grad, fns)

        @jit
        def alpha_loss_agent(params, obs):
            non_grad = (self.H_target, self.pi)
            fns = self.pi_fn
            return alpha_loss(params, obs, non_grad, fns)

        q_loss_val, q_grad = batch_grad(self.q, q_loss_agent, (S, S_n, R, A, eps))

        pi_loss_val, pi_grad = batch_grad(self.pi, pi_loss_agent, (S, eps))

        alpha_loss_val, alpha_grad = batch_grad(self.alpha, alpha_loss_agent, (S, eps))

        # TODO look into options beyond simple gradient descent
        self.q = stu.add_gradient(self.q, q_grad, -LR)
        self.q_targ = stu.interpolate_networks(self.q_targ, self.q, self.tau)
        self.pi = stu.add_gradient(self.pi, pi_grad, -LR)
        self.alpha = self.alpha - LR * alpha_grad

        return q_loss_val, pi_loss_val, alpha_loss_val

    def get_SAC_params(self):
        return SAC_params(
            pi=self.pi,
            q=self.q,
            q_targ=self.q_targ,
            alpha=self.alpha,
            gamma=self.gamma,
            H_target=self.H_target,
        )

    def get_SAC_fns(self):
        return SAC_fns(pi=self.pi_fn, q=self.q_fn)

    def update_2(self, LR=None):
        if LR is None:
            LR = self.LR

        S, Sn, A, R, _ = self.memory_train.sample_batch()

        eps = self.new_eps(A.shape)
        batch = SAC_obs(S, A, R, Sn, eps)
        non_grad = self.get_SAC_params()
        fns = self.get_SAC_fns()

        q_loss_val, q_grad = batch_grad_2(q_loss, self.q, batch, non_grad, fns)

        pi_loss_val, pi_grad = batch_grad_2(pi_loss, self.pi, batch, non_grad, fns)

        alpha_loss_val, alpha_grad = batch_grad_2(
            alpha_loss, self.alpha, batch, non_grad, fns
        )

        # TODO look into options beyond simple gradient descent
        self.q = stu.add_gradient(self.q, q_grad, -LR)
        self.q_targ = stu.interpolate_networks(self.q_targ, self.q, self.tau)
        self.pi = stu.add_gradient(self.pi, pi_grad, -LR)
        self.alpha = self.alpha - LR * alpha_grad

        return q_loss_val, pi_loss_val, alpha_loss_val

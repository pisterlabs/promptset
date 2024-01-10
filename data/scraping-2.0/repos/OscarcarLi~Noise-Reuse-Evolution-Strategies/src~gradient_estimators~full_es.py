"""FullES, the vanilla offline ES method."""
import functools
from typing import Mapping, Sequence, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
from src.utils import profile
from src.utils import tree_utils
from src.utils import common
from src.outer_trainers import gradient_learner
from src.task_parallelization import truncated_step
import chex
import functools

import ipdb


PRNGKey = jax.Array
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))


class FullES(gradient_learner.GradientEstimator):
  """The FullES gradient estimator."""

  def __init__(
      self,
      truncated_step: truncated_step.TruncatedStep,
      std: float,
      T: int,
      loss_normalize: bool = False,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      std: the standard deviation of per-coordinate gaussian noise
      T: the horizon length
      loss_normalize: whether to apply heuristic trick to normalize losses
    """
    self.truncated_step = truncated_step
    self.std = std
    self.T = T
    self.loss_normalize = loss_normalize
    self._total_env_steps_used = 0


  def grad_est_name(self):
    name_list = [
      "FullES",
      f"N={self.truncated_step.num_tasks},K=T,W=T,sigma={self.std}"
    ]
    if self.loss_normalize:
      name_list.insert(1, "loss_normalized")
    return "_".join(name_list)

  def task_name(self):
    return self.truncated_step.task_name()

  @property
  def total_env_steps_used(self,):
    return self._total_env_steps_used

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey):
    
    return None

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state, # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jax.Array]]:
    data_sequence = self.truncated_step.get_batch(steps=self.T)
    # mean_loss, g = \
    (L_pos, L_neg), L_std, g = \
      self.full_es(
        theta=worker_weights.theta,
        key=key,
        data_sequence=data_sequence,
        outer_state=worker_weights.outer_state,
      )
    self._total_env_steps_used += self.T * 2 * self.truncated_step.num_tasks
    # import ipdb; ipdb.set_trace()
    mean_loss = jnp.mean(L_pos + L_neg, axis=[0, 1]) / 2

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=g,
        unroll_state=state,
        unroll_info=None)

    if with_summary:
      if L_std is not None:
        return output, {"mean||loss_std": L_std}
    return output, {}

  @functools.partial(
      jax.jit, static_argnums=(0,))
  def full_es(self,
      theta: MetaParams, # there is a single copy of theta
      key: chex.PRNGKey,
      data_sequence: Any,
      outer_state: Any
    ):
    # key will be used in L_all_single
    key, eps_key = jax.random.split(key, 2)
    keys = jax.random.split(eps_key, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(theta, keys, self.std)

    def L_pos_neg_single(theta):
      """
      Args:
          thetas:
      returns:
          a list of losses obtained by unrolling the same theta for each time step
      """
      init_key, unroll_key = jax.random.split(key, 2)

      pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
      neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, epsilons)

      pos_init_unroll_state = self.truncated_step.init_step_state(
          pos_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),
          theta_is_vector=True)
      neg_init_unroll_state = self.truncated_step.init_step_state(
          neg_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),
          theta_is_vector=True)
      
      def step(scan_state, data):
          pos_inner_unroll_state, neg_inner_unroll_state, t, key = scan_state
          key1, key = jax.random.split(key, num=2)
          
          new_pos_inner_unroll_state, pos_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=pos_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=pos_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),
                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_neg_inner_unroll_state, neg_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=neg_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=neg_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),
                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_scan_state = \
            (new_pos_inner_unroll_state, new_neg_inner_unroll_state, t+1, key)
          return new_scan_state, (pos_inner_unroll_out.loss, neg_inner_unroll_out.loss, new_pos_inner_unroll_state, new_neg_inner_unroll_state)
          
      _, (L_pos, L_neg, state_pos, state_neg) = \
        jax.lax.scan(step,
                    (pos_init_unroll_state, neg_init_unroll_state, 0, unroll_key), data_sequence)
      return (L_pos, L_neg, state_pos, state_neg)

    # L_pos and L_neg should be of shape (T, num_tasks)
    # L_pos, L_neg = L_pos_neg_single(theta)
    (L_pos, L_neg, state_pos, state_neg) = L_pos_neg_single(theta)
    # avg_loss = jnp.mean(L_pos + L_neg, axis=[0, 1]) / 2
    avg_L_pos = jnp.mean(L_pos, axis=0) # of shape (num_tasks,)
    avg_L_neg = jnp.mean(L_neg, axis=0) # of shape (num_tasks,)

    if self.loss_normalize:
      # maybe we should use sum here in combination with Paul's hyperparameter?
      L_std = jnp.std(jnp.concatenate([avg_L_pos, avg_L_neg], axis=0), axis=0)
      multiplier = (avg_L_pos - avg_L_neg) / (2 * self.std * L_std) # (num_tasks,)
    else:
      L_std = None
      multiplier = (avg_L_pos - avg_L_neg) * 1 / (2 * self.std **2)

    es_gradient = jax.tree_util.tree_map(
        lambda eps: jnp.mean(
            eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1] * len(eps.shape[1:])),
            axis=0,
        ),
        epsilons,
    )

    # return avg_loss, es_gradient
    # import ipdb; ipdb.set_trace()
    return (L_pos, L_neg), L_std, es_gradient


class FullES_unjitted(FullES):
  def grad_est_name(self):
    name_list = [
      "FullES_unjitted",
      f"N={self.truncated_step.num_tasks},K=T,W=T,sigma={self.std}"
    ]
    if self.loss_normalize:
      name_list.insert(1, "loss_normalized")
    return "_".join(name_list)
  
  def full_es(self,
      theta: MetaParams, # there is a single copy of theta
      key: chex.PRNGKey,
      data_sequence: Any,
      outer_state: Any
    ):
    # ipdb.set_trace()
    # we don't jit this function because the truncated_step is not jittable
    # key will be used in L_all_single
    key, eps_key = jax.random.split(key, 2)
    keys = jax.random.split(eps_key, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(theta, keys, self.std)

    def L_pos_neg_single(theta):
      """
      Args:
          thetas:
      returns:
          a list of losses obtained by unrolling the same theta for each time step
      """
      init_key, unroll_key = jax.random.split(key, 2)

      pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
      neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, epsilons)
      # ipdb.set_trace()
    
      pos_init_unroll_state = self.truncated_step.init_step_state(
          pos_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),
          theta_is_vector=True)
      neg_init_unroll_state = self.truncated_step.init_step_state(
          neg_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),
          theta_is_vector=True)
      
      
      L_pos = []
      L_neg = []
      def step(scan_state, data):
          pos_inner_unroll_state, neg_inner_unroll_state, t, key = scan_state
          key1, key = jax.random.split(key, num=2)
          
          new_pos_inner_unroll_state, pos_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=pos_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=pos_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),
                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_neg_inner_unroll_state, neg_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=neg_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=neg_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_scan_state = \
            (new_pos_inner_unroll_state, new_neg_inner_unroll_state, t+1, key)
          return new_scan_state, (pos_inner_unroll_out.loss, neg_inner_unroll_out.loss)
      
      carry = (pos_init_unroll_state, neg_init_unroll_state, 0, unroll_key)
      for i in range(tree_utils.first_dim(data_sequence)):
        # import ipdb; ipdb.set_trace()
        data_batch = jax.tree_util.tree_map(lambda x: x[i], data_sequence)
        carry, (L_pos_step, L_neg_step) = step(carry, data_batch)
        L_pos.append(L_pos_step)
        L_neg.append(L_neg_step)

      L_pos = jnp.stack(L_pos, axis=0)
      L_neg = jnp.stack(L_neg, axis=0)
      
      return L_pos, L_neg
      
    # L_pos and L_neg should be of shape (T, num_tasks)
    # L_pos, L_neg = L_pos_neg_single(theta)
    L_pos, L_neg = L_pos_neg_single(theta)
    avg_L_pos = jnp.mean(L_pos, axis=0) # of shape (num_tasks,)
    avg_L_neg = jnp.mean(L_neg, axis=0) # of shape (num_tasks,)

    if self.loss_normalize:
      # maybe we should use sum here in combination with Paul's hyperparameter?
      L_std = jnp.std(jnp.concatenate([avg_L_pos, avg_L_neg], axis=0), axis=0)
      multiplier = (avg_L_pos - avg_L_neg) / (2 * self.std * L_std) # (num_tasks,)
    else:
      L_std = None
      multiplier = (avg_L_pos - avg_L_neg) * 1 / (2 * self.std **2)

    es_gradient = jax.tree_util.tree_map(
        lambda eps: jnp.mean(
            eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1] * len(eps.shape[1:])),
            axis=0,
        ),
        epsilons,
    )

    # return avg_loss, es_gradient
    # import ipdb; ipdb.set_trace()
    return (L_pos, L_neg), L_std, es_gradient
  

class FullES_trajopt(FullES):
  @property
  def grad_est_name(self):
    return \
      ("FullES_trajopt"
      f"_N={self.truncated_step.num_tasks},K=T,W=T,sigma={self.std}")

  @staticmethod
  @functools.partial(jax.vmap, in_axes=(0, None))
  # by doing the vmap, we get an output of shape (M, T)
  def one_zero_convert(inner_step, T):
      # inner_step is an integer (before the vmap)
      # this works for a single inner_step
      # return a vector with the indices up to and include inner_step
      # set to 1 and the rest set to 0
      init_val = (jnp.zeros(T), 0)
      def true_fun():
        return jax.lax.while_loop(
            cond_fun=lambda x: x[1] < inner_step,
            body_fun=lambda x: (x[0].at[x[1]].set(1), x[1] + 1),
            init_val=init_val)[0]
      def false_fun():
        # if inner_step is 0, then we have a reset step and we
        # want to treat it as inner_step is T
        # and thus return a vector of all ones
        return jnp.ones(T)
      return jax.lax.cond(inner_step != 0, true_fun, false_fun)
  
  @staticmethod
  def one_particle_gradient(multiplier, inner_steps, epsilons,):
    """
    multiplier (M,)
    inner_steps (M,)
    epsilons (T, d)

    when we call this function for FullES_trajopt, we have M=T
    """
    T = epsilons.shape[0]
    one_zero_matrix = FullES_trajopt.one_zero_convert(inner_steps, T).T # (T, M)
    multiplier_cumulative = jnp.dot(one_zero_matrix / one_zero_matrix.shape[1],
    # here we need to divide by the number of steps used by the particle (M)
                                    multiplier) # (T,)
    return jnp.multiply(epsilons,
                        jnp.reshape(multiplier_cumulative, multiplier_cumulative.shape + (1,))) # (T, d)
  
  @staticmethod
  def multi_particle_gradient(multiplier, inner_steps, epsilons):
    """
    assume there are N particles, the vmap is ove this particle dimension
    here the multiplier of shape (M, N)
    inner_steps of shape (M, N)
    epsilons of shape (N, T, d)
    """
    return jax.vmap(FullES_trajopt.one_particle_gradient, in_axes=(1, 1, 0))(
      multiplier, inner_steps, epsilons)

  @functools.partial(
      jax.jit, static_argnums=(0,))
  def full_es(self,
      theta: MetaParams, # there is a single copy of theta
      key: chex.PRNGKey,
      data_sequence: Any,
      outer_state: Any
    ):
    # key will be used in L_all_single
    key, eps_key = jax.random.split(key, 2)
    keys = jax.random.split(eps_key, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(theta, keys, self.std)

    def L_pos_neg_single(theta):
      """
      Args:
          thetas:
      returns:
          a list of losses obtained by unrolling the same theta for each time step
      """
      init_key, unroll_key = jax.random.split(key, 2)

      pos_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) + b, theta, epsilons)
      neg_perturbed_thetas = jax.tree_util.tree_map(lambda a,b: jnp.expand_dims(a, 0) - b, theta, epsilons)
    
      pos_init_unroll_state = self.truncated_step.init_step_state(
          pos_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),
          theta_is_vector=True)
      neg_init_unroll_state = self.truncated_step.init_step_state(
          neg_perturbed_thetas,
          outer_state,
          jax.random.split(init_key, self.truncated_step.num_tasks),          theta_is_vector=True)
      
      def step(scan_state, data):
          pos_inner_unroll_state, neg_inner_unroll_state, t, key = scan_state
          key1, key = jax.random.split(key, num=2)
          
          new_pos_inner_unroll_state, pos_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=pos_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=pos_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),
                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_neg_inner_unroll_state, neg_inner_unroll_out = \
            self.truncated_step.unroll_step(
                  theta=neg_perturbed_thetas, # pick the theta at the time step t.
                  unroll_state=neg_inner_unroll_state,
                  key_list=jax.random.split(key1, self.truncated_step.num_tasks),                  data=data,
                  outer_state=outer_state,
                  theta_is_vector=True)
          new_scan_state = \
            (new_pos_inner_unroll_state, new_neg_inner_unroll_state, t+1, key)
          return new_scan_state, (pos_inner_unroll_out.loss, neg_inner_unroll_out.loss, new_pos_inner_unroll_state, new_neg_inner_unroll_state)
          
      _, (L_pos, L_neg, state_pos, state_neg) = \
        jax.lax.scan(step,
                    (pos_init_unroll_state, neg_init_unroll_state, 0, unroll_key), data_sequence)
      return (L_pos, L_neg, state_pos, state_neg)

    # L_pos and L_neg should be of shape (T, num_tasks)
    # L_pos, L_neg = L_pos_neg_single(theta)
    (L_pos, L_neg, state_pos, state_neg) = L_pos_neg_single(theta)

    multiplier = (L_pos - L_neg) / (2 * self.std **2) # (T, N)
    es_gradient_each_particle = jax.tree_util.tree_map(
      lambda eps: FullES_trajopt.multi_particle_gradient(
        multiplier, state_pos.inner_step, eps),
      epsilons,
    ) # (N, T, d)
    es_gradient = jax.tree_util.tree_map(
      lambda x: jnp.mean(x, axis=0), es_gradient_each_particle) # (T, d)

    # return avg_loss, es_gradient
    # the second element is for loss standard deviation
    return (L_pos, L_neg), None, es_gradient




if __name__ == "__main__":
  import haiku
  import gymnasium as gym
  from src.task_parallelization import openai_gym_truncated_step

  T = 1000
  horizon_length = T
  env_name = "Swimmer-v4"
  env = gym.make(env_name)
  policy = haiku.transform(
    lambda x: haiku.Linear(env.action_space.shape[0], with_bias=False, w_init=haiku.initializers.Constant(0.0))(x))  # type: ignore

  truncated_step = \
    openai_gym_truncated_step.OpenAIGymTruncatedStep(
      env_name=env_name,
      num_tasks=10,
      T=T,
      policy=policy,
      random_initial_iteration=False,
      truncation_window_size=T,
      # for full trajectory methods like FullES and FullGradient
      # the random_initial_iteration will be turned off so even though
      # truncation_window_size is set to arbitrary values it doesn't matter
    )


  fulles = FullES_unjitted(
    truncated_step=truncated_step,
    std=0.3,
    T=T, loss_normalize=False,)
  print(fulles.grad_est_name)

  fulles = FullES(
    truncated_step=truncated_step,
    std=0.3,
    T=T, loss_normalize=True,)
  print(fulles.grad_est_name)
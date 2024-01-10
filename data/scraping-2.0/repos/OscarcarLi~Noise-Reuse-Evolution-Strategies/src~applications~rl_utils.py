from absl import flags

import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku

from src.utils import common
from src.task_parallelization import openai_gym_truncated_step
from src.gradient_estimators import gradient_estimator_utils

FLAGS = flags.FLAGS

# problem type
flags.DEFINE_string("gym_env_name", None, "name of the gym environment to use")
flags.DEFINE_string("rl_policy_architecture", "linear", "architecture of the policy")
flags.DEFINE_string("rl_init_type", "zero",
                    ("initialization type for the policy"
                    "zero: zero init"
                    "normal: fan_in truncated normal init"))
flags.DEFINE_float("rl_init_scale", 0.03, 
                  ("variance scale for the init distribution"
                  "not used if rl_init_type is zero"))
flags.DEFINE_string("rl_model_path", None, "path to the model to load")
flags.DEFINE_integer("rl_model_iteration", None, "iteration of the model to load")

NUM_PARTICLES_EVALUATION = 20

def use_immutable_state_in_truncated_state():
  ESTIMATORS_THAT_NEED_IMMUTABLE = ["TruncatedESBiased"]
  # faster using this
  ENVS_THAT_NEED_IMMUTABLE = []
  a = FLAGS.gradient_estimator.upper() in [x.upper() for x in ESTIMATORS_THAT_NEED_IMMUTABLE]
  b = FLAGS.gym_env_name.upper() in [x.upper() for x in ENVS_THAT_NEED_IMMUTABLE]
  return a or b


def create_task_train_test():
  # to work with the interface of main training, this function in fact directly returns the truncated_step_train and truncated_step_test, this is fine for now as truncated_step_train includes the interface.

  # we need to the dimensions of the state and action space
  use_immutable = use_immutable_state_in_truncated_state()
  env = gym.make(FLAGS.gym_env_name)
  if FLAGS.rl_policy_architecture.upper() == "LINEAR":
    # we learn a linear policy without bias
    if FLAGS.rl_init_type.upper() == "ZERO":
      w_init = haiku.initializers.Constant(0.0)
    elif FLAGS.rl_init_type.upper() == "NORMAL":
      w_init = haiku.initializers.VarianceScaling(
        scale=FLAGS.rl_init_scale, mode='fan_in', distribution='truncated_normal')
    else:
      assert False, "unknown rl_init_type"

    policy = haiku.transform(
      lambda x: haiku.Linear(
                  env.action_space.shape[0], with_bias=False,  # type: ignore
                  w_init=w_init,
                )(x))
  elif FLAGS.rl_policy_architecture.upper() == "MLP":
    policy = haiku.transform(
      lambda x: haiku.nets.MLP(
                          output_sizes=[30, env.action_space.shape[0]],  # type: ignore
                          w_init=haiku.initializers.VarianceScaling(
                          scale=0.3, mode='fan_in', distribution='truncated_normal'),
                          # w_init=haiku.initializers.Constant(0.0),
                          # b_init=haiku.initializers.Constant(0.0),
                          activation=jax.nn.relu,
        )(x))
  else:
    raise ValueError("unknown rl_policy_architecture")

  if FLAGS.rl_model_path is not None and FLAGS.rl_model_iteration is not None:
    theta_template = policy.init(jax.random.PRNGKey(0), jnp.zeros(env.observation_space.shape[0]))  # type: ignore
    init_theta = common.load_theta(
                            theta_template=theta_template,
                            step=FLAGS.rl_model_iteration,
                            saved_path=FLAGS.rl_model_path)
  else:
    init_theta = None
  random_initial_iteration_train = \
    gradient_estimator_utils.use_random_initial_iteration_for_truncated_step_train()
  truncated_step_train = openai_gym_truncated_step.OpenAIGymTruncatedStep(
      env_name=FLAGS.gym_env_name,
      num_tasks=(FLAGS.num_particles // FLAGS.num_gradient_estimators),
      policy=policy,
      T=FLAGS.horizon_length,
      random_initial_iteration=random_initial_iteration_train,
      truncation_window_size=FLAGS.trunc_length,
      init_theta=init_theta, # we don't pass in a fixed theta for meta-init)
      immutable_state=use_immutable,
  )
  truncated_step_test = openai_gym_truncated_step.OpenAIGymTruncatedStep(
    env_name=FLAGS.gym_env_name,
    num_tasks=NUM_PARTICLES_EVALUATION,
    policy=policy,
    T=FLAGS.horizon_length,
    random_initial_iteration=False,
    truncation_window_size=FLAGS.trunc_length,
    init_theta=None, # for testing the init function is never used
    immutable_state=use_immutable,)

  return truncated_step_train, truncated_step_test

def create_truncated_step_train_test(train_task, test_task):
  assert isinstance(train_task, openai_gym_truncated_step.OpenAIGymTruncatedStep), "train_task should be an instance of OpenAIGymTruncatedStep"
  assert isinstance(test_task, openai_gym_truncated_step.OpenAIGymTruncatedStep), "test_task should be an instance of OpenAIGymTruncatedStep"

  return train_task, test_task

def create_truncated_step_train_for_evaluation(train_task):
  # train task is actually an openai openai_gym_truncated_step.OpenAIGymTruncatedStep object
  use_immutable = use_immutable_state_in_truncated_state()
  truncated_step_train_evaluation = \
    openai_gym_truncated_step.OpenAIGymTruncatedStep(
      env_name=FLAGS.gym_env_name,
      num_tasks=NUM_PARTICLES_EVALUATION,
      policy=train_task.policy,
      T=FLAGS.horizon_length,
      random_initial_iteration=False,
      truncation_window_size=FLAGS.trunc_length,
      init_theta=None, # we don't pass in a fixed theta for meta-init)
      immutable_state=use_immutable,)
  return truncated_step_train_evaluation
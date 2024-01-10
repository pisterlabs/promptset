#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs


def Register_OpenAI_Ros_Env(task_env, max_episode_steps_per_episode=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = True

    # Cubli Moving Cube
    if task_env == 'MavDrone-v0':

        register(
            id=task_env,
            entry_point='openai_ros:task_envs.mav_drone.mav_drone_follow.MavDroneFollowEnv',
            max_episode_steps=max_episode_steps_per_episode,
        )

        # import our training environment
        from openai_ros.task_envs.mav_drone import mav_drone_follow

    # Add here your Task Envs to be registered
    else:
        result = False

    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = [env_spec.id for env_spec in envs.registry.all()]
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + str(task_env)

    return result

#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs
import gym
import rospy

def RegisterOpenAI_Ros_Env(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    result = True
    if task_env in gym.envs.registry.env_specs:
        rospy.logwarn("Environment already registered, . . .\n re-registering. . .")
        del gym.envs.registry.env_specs[task_env]
        
    if task_env == 'UR5LabSimTask-v0':
        # We have to import the Class that we registered so that it can be found afterwards in the Make
        from openai_ros.task_envs.ur5_lab import sim_task_v0
        # We register the Class through the Gym system
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.ur5_lab.sim_task_v0:UR5LabSimTask',
            max_episode_steps=max_episode_steps,
        )
        
    elif task_env == 'UR5LabRealTask-v0':
        # We have to import the Class that we registered so that it can be found afterwards in the Make
        from openai_ros.task_envs.ur5_lab import real_task_v0
        # We register the Class through the Gym system
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.ur5_lab.real_task_v0:UR5LabRealTask',
            max_episode_steps=max_episode_steps,
        )
        
    else:
        result = False

    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        #print("REGISTERED GYM ENVS===>"+str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result


def GetAllRegisteredGymEnvs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids

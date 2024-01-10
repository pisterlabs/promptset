#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs


def RegisterOpenAI_Ros_Env(task_env):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = True

    if task_env == 'DeepFollow-v0':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation

    elif task_env == 'DeepFollow-v1':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_with_joints:DeepFollowEnv'
        )
        # import our training environment --- check if import formation or formation_With_joints
        from openai_ros.task_envs.firefly import formation_with_joints

    elif task_env == 'DeepFollow-v2':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_multiagent_joints:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_multiagent_joints


    elif task_env == 'DeepFollow-v3':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_yawrate:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_yawrate


    elif task_env == 'DeepFollow-v4':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_end2end:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_end2end

    elif task_env == 'DeepFollow-v5':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_sac:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_sac

    elif task_env == 'DeepFollow-v6':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_multiagent_gt_joints:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_multiagent_gt_joints


    elif task_env == 'DeepFollow-v7':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_multiagent_gt_inf:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_multiagent_gt_joints

    elif task_env == 'DeepFollow-v8':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_singleagent_gt:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_singleagent_gt

    elif task_env == 'DeepFollow-v9':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_singleagent_yawrate_gt:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_singleagent_yawrate_gt

    elif task_env == 'DeepFollow-v10':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_ego_yawrate_gt:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_ego_yawrate_gt

    elif task_env == 'DeepFollow-v11':
        print('Entering:'+task_env)
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.firefly.formation_ego_yawrate_hmr_gt:DeepFollowEnv'
        )
        # import our training environment
        from openai_ros.task_envs.firefly import formation_ego_yawrate_hmr_gt

    # Add here your Task Envs to be registered
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

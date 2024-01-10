# The code is based on the ACKTR implementation from OpenAI baselines
# This code implements method by Kangin & Pugeault "On-Policy Trust Region Policy Optimisation with Replay Buffers"
def start(fold, env_id):
    from baselines import logger
    from baselines.common.cmd_util import mujoco_arg_parser
    from baselines.trpo_replay.acktr_cont import train
    from algorithm_parameters import algorithm_parameters
    import os
    import tensorflow as tf
    tf.reset_default_graph()
    os.environ ['OPENAI_LOGDIR'] = 'logs_' + env_id + '_' + str(fold)
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    parameters = algorithm_parameters()
    train(env_id, parameters=parameters, seed=args.seed)

if __name__ == "__main__":
    #env_ids = {'Reacher-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Ant-v2', 'Humanoid-v2', 'Hopper-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2'}
#    env_ids = {'Ant-v2'}#{'Walker2d-v2', 'HalfCheetah-v2', 'Swimmer-v2'}
    env_ids = {'Ant-v2'}
    #env_ids = {'MountainCarContinuous-v0'}
    NUM_FOLDS = 3
    for env_id in env_ids:
        for i in range(NUM_FOLDS):
            start(i, env_id)

# # FAPS PLMAgents
# ## FAPS PLM ML-Agent Learning

import docopt
import logging
import os
from OpenAIGym.trainer_controller import TrainerController

if __name__ == '__main__':
    logger = logging.getLogger("FAPSPLMAgents")

    _USAGE = '''
    Usage:
      main [options]
      main --help

    Options:
      --brain_name=<path>        Name of the brain to use. [default: DQN]. 
      --environment=<env>        Name of the environment to use. [default: 'eflex-agent-v0'].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: True].
      --render                   Whether the environment should be rendered or not [default: False].
      --save-freq=<n>            Frequency at which to save model [default: 10000].
      --seed=<n>                 Random seed used for training [default: -1].
      --train                    Whether to train model, or only run inference [default: False].
      --use-gpu                  Make use of GPU.
    '''
    options = None
    try:
        options = docopt.docopt(_USAGE)

    except docopt.DocoptExit as e:
        # The DocoptExit is thrown when the args do not match.
        # We print a message to the user and the usage block.

        print('Invalid Command!')
        print(e)
        exit(1)

    # General parameters
    brain_name = options['--brain_name']
    environment = options['--environment']

    seed = int(options['--seed'])
    load_model = bool(options['--load'])
    train_model = bool(options['--train'])
    save_freq = int(options['--save-freq'])
    keep_checkpoints = int(options['--keep-checkpoints'])

    lesson = int(options['--lesson'])
    use_gpu = int(options['--use-gpu'])
    render = bool(options['--render'])

    # log the configuration
    # logger.info(options)

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    tc = TrainerController(use_gpu, [brain_name], [environment],
                           render, save_freq, load_model, train_model,
                           keep_checkpoints, seed, TRAINER_CONFIG_PATH)
    tc.start_learning()
    exit(0)

# import sys
# import time
# import logging
# import gym
# import gym_eflex_agent
#
#
# sys.path.insert(0, "..")
#
# try:
#     from IPython import embed
# except ImportError:
#     import code
#
#     def embed():
#         vars = globals()
#         vars.update(locals())
#         shell = code.InteractiveConsole(vars)
#         shell.interact()
#
# interactive = True
#
# if __name__ == "__main__":
#     # optional: setup logging
#     logging.basicConfig(level=logging.WARN)
#     logger = logging.getLogger("opcua.address_space")
#     logger.setLevel(logging.DEBUG)
#
#     # test gym
#     env = gym.make('eflex-agent-v0')
#     observation = env.reset()
#     for _ in range(5000):
#         observation, reward, done, info = env.step(env.action_space.sample())
#         print(info['info'])
#         env.render('human')
#     env.close()
#
#     # try:
#     #     if interactive:
#     #         embed()
#     #     else:
#     #         while True:
#     #             time.sleep(0.5)
#     #
#     # except IOError:
#     #     pass
#     # finally:
#     #     print("done")

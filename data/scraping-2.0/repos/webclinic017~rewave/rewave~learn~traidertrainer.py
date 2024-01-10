from rewave.environment.long_portfolio import PortfolioEnv
from rewave.wrappers import SoftmaxActions, ConcatStates

from tensorforce.contrib.openai_gym import OpenAIGym


class TFOpenAIGymCust(OpenAIGym):
    def __init__(self, gym_id, gym):
        self.gym_id = gym_id
        self.gym = gym
        self.visualize = False


class TraderTrainer:
    def __init__(self, config, fake_data=False, restore_dir=None, save_path=None, device="cpu",
                 agent=None):
        """
        :param config: config dictionary
        :param fake_data: if True will use data generated randomly
        :param restore_dir: path to the model trained before
        :param save_path: path to save the model
        :param device: the device used to train the network
        :param agent: the nnagent object. If this is provides, the trainer will not
        create a new agent by itself. Therefore the restore_dir will not affect anything.
        """




env = PortfolioEnv(
    df=df_train,
    steps=40,
    scale=True,
    trading_cost=0.0025,
    window_length = window_length,
    output_mode='EIIE',
#     random_reset=False,
)
# wrap it in a few wrappers
env = ConcatStates(env)
env = SoftmaxActions(env)
environment = TFOpenAIGymCust('CryptoPortfolioEIIE-v0', env)

# sanity check out environment is working
state = environment.reset()
state, reward, done=environment.execute(env.action_space.sample())
state.shape

# sanity check out environment is working
state = env.reset()
state, reward, done, info=env.step(env.action_space.sample())
state.shape
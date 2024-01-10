from reinforch.agents import PolicyGradientAgent
from reinforch.core.memorys import PGMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner


def test_pg():
    gym_id = 'CartPole-v0'

    env = OpenAIGym(gym_id)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    memory = PGMemory()

    agent = PolicyGradientAgent(n_s=n_s,
                                n_a=n_a,
                                memory=memory,
                                config='tests/configs/test_pg.json')
    with Runner(agent=agent,
                environment=env,
                verbose=False) as runner:
        runner.train(total_episode=10,
                     max_step_in_one_episode=200,
                     save_model=False,
                     save_final_model=False,
                     visualize=False)


def test_pg_continous():
    gym_id = 'MountainCarContinuous-v0'

    env = OpenAIGym(gym_id)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    action_dim = env.actions.get('max_value')
    memory = PGMemory()

    agent = PolicyGradientAgent(n_s=n_s,
                                n_a=n_a,
                                action_dim=action_dim,
                                memory=memory,
                                config='tests/configs/test_pg_continous.json')
    with Runner(agent=agent,
                environment=env,
                verbose=False) as runner:
        runner.train(total_episode=1,
                     max_step_in_one_episode=200,
                     save_model=False,
                     save_final_model=False,
                     visualize=False)

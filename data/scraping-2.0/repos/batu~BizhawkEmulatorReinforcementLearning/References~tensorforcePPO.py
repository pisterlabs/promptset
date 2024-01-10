import gym_bizhawk
import os
import time

from support_utils import save_hyperparameters, send_email, visualize_cumulative_reward, visualize_max_reward

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

REPLAY = False
run_number = 21
experiment_name = "Tensorforce"

TB_path = f"Results/TensorBoard/{experiment_name}/"
try:
    os.mkdir(TB_path[:-1])
except:
    pass

try:
    os.mkdir(f"{TB_path}README")
except:
    pass

models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v1'

changes = """Switched to tensorforce."""
reasoning = """I found the documentation lacking and I wasnt trusting in the details."""
hypothesis = """Now gradient over policies. Hopefully model free will help"""

if len(hypothesis) + len(changes) + len(reasoning) < 140:
    print("NOT ENOUGH LOGGING INFO. \n PLEASE WRITE SOME MORE DETAILS.")
    exit()

with open(f"{TB_path}/README/README.txt", "w") as readme:
    start_time_ascii = time.asctime(time.localtime(time.time()))
    algorithm = os.path.basename(__file__)[:-2]
    print(f"Experiment start time: {start_time_ascii}", file=readme)
    print(f"\nAlgorithm:\n{algorithm}", file=readme)
    print(f"\nThe Changes:\n{changes}", file=readme)
    print(f"\nReasoning:\n{reasoning}", file=readme)
    print(f"\nHypothesis:\n{hypothesis}", file=readme)
    print(f"\nResults:\n", file=readme)

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
environment = OpenAIGym('Bizhawk-v1', visualize=False)
env = gym_bizhawk.BizHawk(logging_folder_path=TB_path)
# env = OpenAIGym('CartPole-v0', visualize=False)

nb_actions = env.action_space.n
# BREADCRUMBS_START
window_length = 1
# BREADCRUMBS_END

# Sequential Model
#print(env.observation_space.shape)

# BREADCRUMBS_START
network_spec = [
    # dict(type='embedding', indices=100, size=32),
    dict(type='flatten'),
    dict(type='dense', size=32),
    dict(type='dense', size=16),
]
# BREADCRUMBS_END

states = {'shape': env.observation_space.shape,
           'type': 'float'}
actions = {'type': 'float',
           'num_actions': nb_actions}
# BREADCRUMBS_START
agent = PPOAgent(
    states=states,
    actions=actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=20,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

runner = Runner(agent=agent, environment=env)
# BREADCRUMBS_END

folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
env.run_name = run_name

os.mkdir(run_path)

# with open(f"{run_path}/run_summary.txt", "w") as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
#
# with open(f"{run_path}/config.json", "w") as outfile:
#     json_string = model.to_json()
#     json.dump(json_string, outfile)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["DQN.py", "gym_bizhawk.py"], f"{run_path}/run_summary.txt")

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")

# BREADCRUMBS_START
runner.run(episodes=3000, max_episode_timesteps=200)
runner.close()
# BREADCRUMBS_END


total_run_time = round(time.time() - start_time, 2)
print("Training is done.")

visualize_cumulative_reward(input_file=f"{run_path}/cumulative_reward.txt",
                            ouput_destionation=f"{run_path}/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

visualize_max_reward(input_file=f"{run_path}/max_reward.txt",
                            ouput_destionation=f"{run_path}/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .",
            run_path=run_path, experiment_name=experiment_name, run_number=folder_count)

env.shut_down_bizhawk_game()
# Finally, evaluate our algorithm for 5 episodes.
# movie.save("C:/Users/user/Desktop/VideoGame Ret/RL Retrieval/movie")
# dqn.test(env, nb_episodes=1, visualize=False)

time.sleep(5)

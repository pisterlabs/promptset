import gym
import textworld.gym

from llamp.openai_agent import OpenAIAgent
from llamp.orca2_agent import Orca2Agent
from llamp.mistral_orca_agent import MistralOrcaAgent
from llamp.minichat_agent import MiniChatAgent
from llamp.yi_agent import YiAgent
from llamp.human_agent import HumanAgent
from llamp import utils

import sys
import os


# import warnings
# warnings.filterwarnings("ignore")

if __name__=="__main__":

    ####################
    # SET THE GLOBAL PARAMS
    ####################
    AGENT_TYPE = ""
    MAX_MOVES = 50
    GAME_TYPE = "custom"

    if sys.argv:
        print("Called TW script with:")
        print(sys.argv)
        AGENT_TYPE=sys.argv[1]
    else:
        # AGENT_TYPE = "openai"
        # AGENT_TYPE = "orca2"
        # AGENT_TYPE = "orca_mistral_test"
        # AGENT_TYPE="minichat_test"
        # AGENT_TYPE="minichat"
        AGENT_TYPE="human"


    ##########
    # PARAMS

    if len(sys.argv)>2:
        if sys.argv[2] == "--custom":
            GAME_TYPE = "custom"
            custom_params = sys.argv[3:]
        elif sys.argv[2] == "--simple":
            GAME_TYPE = "simple"
            simple_params = sys.argv[3:]
    else:
        simple_params = ["dense","detailed",1234]
        # custom_params = [2,10,5,1234]
        custom_params = [1,2,2,1234]

    

    ####################
    # Running the Agent
    ####################

    if GAME_TYPE=="simple":
        # simple_params = ["dense","detailed",1234]
        game_path = utils.construct_simple_game_name(*simple_params)
        log_path = utils.construct_simple_game_name(*simple_params, log_path=True)
        # game_header = "It's time to explore the amazing world of TextWorld!"
    elif GAME_TYPE=="custom":
        # custom_params = [2,10,5,1234]
        game_path = utils.construct_custom_game_name(*custom_params)
        log_path = utils.construct_custom_game_name(*custom_params, log_path=True)
        # game_header = "Welcome to TextWorld!"
    else:
        # game_path = os.path.join("games","zork_games/zork1.z5")
        exit()

    ROOT_GAME_LOG = "game_logs"
    # if not os.path.exists(ROOT_GAME_LOG):
    #     os.mkdir(ROOT_GAME_LOG)
    log_path = os.path.join(ROOT_GAME_LOG,log_path)
       

    # Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(game_path,
                                         max_episode_steps=MAX_MOVES)
    env = gym.make(env_id)  # Start the environment.


    # Load the agent you want:
    if AGENT_TYPE=="openai":
        agent = OpenAIAgent()
    elif AGENT_TYPE=="orca2":
        agent = Orca2Agent()
    elif AGENT_TYPE=="orca_mistral":
        agent = MistralOrcaAgent()
    elif AGENT_TYPE=="orca_mistral_test":
        agent = MistralOrcaAgent(test_mode=True)
        agent.act("How are you?")
        agent.save()
        exit()
    elif AGENT_TYPE=="minichat_test":
        agent = MiniChatAgent(test_mode=True)
        agent.act("Yes?")
        agent.save()
        exit()
    elif AGENT_TYPE=="minichat":
        agent=MiniChatAgent()
    elif AGENT_TYPE=="yi":
        agent=YiAgent()
    elif AGENT_TYPE=="human":
        print("You will play with the system")
        agent = HumanAgent()
    else:
        NotImplementedError("This agent is not implemented.")
        exit()


    # New Environment and Observations.
    obs, infos = env.reset()  # Start new episode.
    env.render()

    # cut_header_index = obs.index(game_header)
    cut_header_index = 1211
    first_obs = obs[cut_header_index:]

    command = agent.act(first_obs)
    print(command)

    score, moves, done = 0, 0, False


    try:
        #### MAIN GAME LOOP
        while not done:
            obs, score, done, infos = env.step(command)
            print("="*20)
            env.render()
            print("="*20)

            moves += 1
            if moves % MAX_MOVES == 0:
                done = True
                continue

            if not done:
                command = agent.act(obs)
                print(command)

    finally:
        agent.update_save_path(log_path)
        agent.save()
        env.close()
        print("moves: {}; score: {}".format(moves, score))


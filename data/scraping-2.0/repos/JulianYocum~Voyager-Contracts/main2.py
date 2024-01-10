import argparse
from voyager import MultiAgentVoyager
import time
from api_keys import openai_api_key
from datetime import datetime
import voyager.utils as U

# Argument parser
parser = argparse.ArgumentParser(description='Running Voyager with different sets of parameters.')
parser.add_argument('--port', type=int, default=49172, help='MC port number (default: 49172)')
# parser.add_argument('--server_port', type=int, default=3000, help='Server port number (default: 3000)')
args = parser.parse_args()

mc_port = args.port
options = {
    'azure_login':  None,
    'mc_port': mc_port,
    'openai_api_key': openai_api_key,
    # skill_library_dir=skill_library_dir, # Load a learned skill library.
    # ckpt_dir: ckpt_dir, # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    'resume':False, # Do not resume from a skill library because this is not learning.
    'env_wait_ticks':80,
    # 'env_request_timeout': 600,
    'action_agent_task_max_retries':50,
    'action_agent_show_chat_log':True,
    'action_agent_temperature':0.3,
    'action_agent_model_name': "gpt-4-0613", # #"gpt-4-0613",
    'critic_agent_model_name': "gpt-4-0613", #"gpt-3.5-turbo", #"gpt-4-0613",
}

multi_options = {
    'scenario_file': "cleanup.json",
    'continuous': True,
    'episode_timeout': 120, #120,
    'num_episodes': 1,
    'negotiator_model_name': "gpt-4-0613",
    'negotiator_temperature': 0.7,
    'options': options
}

start_time = time.time()

for contract_i in range(4, 10):

    save_dir = f"saves/cleanup/contract{contract_i}"
    U.f_mkdir(save_dir)

    contract_env = MultiAgentVoyager(
        **multi_options,
        contract_mode = "auto",
        save_dir=f"{save_dir}/contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    contract_env.load_scenario(reset='hard')
    for i in range(3):
        try:
            contract_env.negotiate_contract()
            break
        except:
            print('negotiation failed')
            continue
    if contract_env.contract is None:
        raise Exception('Contract negotiation failed after 3 tries')
    contract = contract_env.contract
    contract_env.close()

    for game in range(5):

        multi_agent = MultiAgentVoyager(
            **multi_options,
            contract_mode = "manual",
            contract=contract,
            save_dir=f"{save_dir}/game{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        multi_agent.run()
        multi_agent.close()

    print(f"Contract {contract} completed. {time.time() - start_time} seconds elapsed.")
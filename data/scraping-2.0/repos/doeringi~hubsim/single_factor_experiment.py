from components.agent.BaseAgent import BaseAgent
from components.common.variant_testing_helper import single_factor_variants
import autogen
import json
import os
import shutil
import time
import openai

config_list = [
    {
        "model": "Mistral-7B-Instruct-v0.1",
        "api_base": "http://localhost:8000/v1",
        "api_type": "open_ai",
        "api_key": "NULL",  # just a placeholder
        "timeout": 1000,
        "max_retries": 9000000
    }
]

llm_config = {"config_list": config_list, "seed": 42}
autogen.ChatCompletion.start_logging()

def save_log_history(content, id):
    with open(f"single-factor-experiments/logs/experiment-log-{id}.json", "w") as file:
        json.dump(content, file)

variants = single_factor_variants()
variants = variants[1]

# run configurations
number_of_experiments = 15
initial_chat_message = "Hello Mister Heine, thanks for inviting me to see the apartment. Let's talk about the rental price."
max_rounds = 6

for index, variant in variants:
    variant_folder = "single-factor-experiments/fifty_square_meters" # + variants[index][0][1]["id"]
    print(variant_folder)
    if not os.path.exists(variant_folder):
            os.makedirs(variant_folder)
            print(f"Folder '{variant_folder} created.")
    print("Starting experiment...")
    for experiment in range(0, number_of_experiments):
        
        max_retries = 200
        retry_delay = 4
        attempt = 0
        while attempt < max_retries:
            try:
                agent = BaseAgent()
                
                renter = autogen.AssistantAgent(name=variants[0][0][1]["name_id"], system_message=variants[1]["renter_system_message"], llm_config=llm_config)
                landlord = autogen.AssistantAgent(name=variants[0][0][0]["name_id"], system_message=variants[1]["landlord_system_message"], llm_config=llm_config)
                
                print("Running experiment...")
                conversation = agent.run_agent_to_agent_conversation(agents=[renter, landlord], max_round=max_rounds, llm_config=llm_config, init_chat_message=initial_chat_message)
                agent.save_conversation(groupchat=conversation, path=variant_folder+ "/" + str(agent.id))
                
                if os.path.exists(".cache"):
                    shutil.rmtree(".cache")
                    
                print(f"Experiment with the id {agent.id} succeeded.")
                break
            except openai.error.Timeout as e:
                print("Timeout Error. Trying again...")
                attempt += 1
                
                if attempt < 200:
                    save_log_history(autogen.ChatCompletion.logged_history, agent.id)
                
                if os.path.exists(".cache"):
                    shutil.rmtree(".cache")
                print(f"Try Counter: {attempt}")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"An error occured: {e}. Trying again.")
                attempt += 1
                if os.path.exists(".cache"):
                    shutil.rmtree(".cache")
                print(f"Try Counter: {attempt}")
                time.sleep(retry_delay)


        
            
from components.experiment.BaseExperiment import BaseExperiment
from components.common.variant_testing_helper import single_factor_variants_renter_name
import autogen
import json
import os
import shutil
import time
import openai
import subprocess
from datetime import datetime

config_list = [
    {
        "model": "Mistral-7B-Instruct-v0.1",
        "base_url": "http://localhost:8000/v1",
        "api_key": "NULL",  # if not needed add NULL as placeholder
    }
]

# set temperature for sampling
llm_config = {"config_list": config_list, 
              "cache_seed": 42, 
              "temperature": 0.6,
#               "timeout": 30,
#               "max_retries": 5
            }

variants = single_factor_variants_renter_name()
variants = variants

# models to use
model_paths= ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", "01-ai/Yi-6B-Chat-8bits"]
model_path = "mistralai/Mistral-7B-Instruct-v0.1"

# experiment metadata for folder creation (here the experiments with different inital_chat_message)
conversation_types = ["no-name-in-start", "name-from-origin-in-start"]
conversation_type = "name-from-origin-in-start"
timestamp = datetime.now()
timestamp = timestamp.strftime("%Y%m%d")

# run configurations
number_of_experiments = 50

max_rounds = 6 # maximum rounds in a conversation, where one round is one reply from a party

# init fastchat
# helper = BaseExperiment()
# helper.start_fastchat(model_path=model_path)

# time.sleep(120) # fastchat needs some time to load the model and get ready

for variant in variants:
    variant_folder = "single-factor-experiments/" + config_list[0]["model"] + "-" + conversation_type + "-" + timestamp + "/" + variant[0][1]["name_id"] # create a folder with an experiment identifier and for each renter name
    
    if not os.path.exists(variant_folder):
            os.makedirs(variant_folder)
            print(f"Folder '{variant_folder} created.")
    print("Starting experiment...")
    for experiment in range(0, number_of_experiments):
        
        max_retries = 100
        retry_delay = 4
        attempt = 0
        while attempt < max_retries:
            try:
                experiment_helper = BaseExperiment()
                initial_chat_message = f"Hello Mister Heine, my name is {variant[0][1]['name_id']}. Thanks for inviting me to see the apartment. Let's talk about the rental price."
                renter = autogen.AssistantAgent(name=variant[0][1]["name_id"], system_message=variant[0][1]["renter_system_message"], llm_config=llm_config)
                landlord = autogen.AssistantAgent(name=variant[0][0]["name_id"], system_message=variant[0][0]["landlord_system_message"], llm_config=llm_config)
                
                print(f"Running experiment: {str(experiment_helper.id)}")
                
                # Note: The conversation is set to round_robin, therefore first speaker is set in agents
                conversation = experiment_helper.run_agent_to_agent_conversation(agents=[renter, landlord], max_round=max_rounds, llm_config=llm_config, init_chat_message=initial_chat_message)
                experiment_helper.save_conversation(groupchat=conversation, path=variant_folder + "/" + str(experiment_helper.id))
                
                if os.path.exists(".cache"):
                    shutil.rmtree(".cache")
                    
                print(f"Experiment with the id {experiment_helper.id} succeeded.")
                break
            except Exception as e:
                print(f"An error occured: {e}. Trying again.")
                attempt += 1
                
                if os.path.exists(".cache"):
                    shutil.rmtree(".cache")
                
                if attempt >= 15:
                    experiment_helper.stop_fastchat()
                    experiment_helper.start_fastchat(model_path)
                
                print(f"Try Counter: {attempt}")
                time.sleep(retry_delay)


        
            
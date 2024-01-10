"""
Script for evaluating how well autogen agents
can carry out tasks given different hyperparameters

NOTE: to kill the wandb process:
>  pkill wandb-service

"""
import functools
import pprint as pp
from pathlib import Path

import wandb
import openai
from dotenv import load_dotenv
load_dotenv()

PROJECT_NAME = "wandb-autogen"
DOC_FILE = "documents/2308.08155.pdf"

# Helper functions
from utils.misc_utils import load_sweep_config
from utils.qa_generation_utils import generate_eval_dataset
from utils.agent_utils import (get_config, instantiate_agents, get_qa_response)
from utils.eval_utils import (run_evaluation_chain, modify_qa_pairs, get_precision_score, get_squad_score)

################################################################################

def objective(config: dict, qa_pairs: list, autogen_logs_filename: str) -> (float, float):
    """Define the objective function to optimize."""
    model_type = config['llm']
    temperature = config['temperature']

    # 1. Set up Agent
    config_list = get_config(model_type, config['dotenv_path'])
    agents = instantiate_agents(config_list, temperature)
    
    # 2. Test output of autogen agents and get costs
    predictions, costs = [], []
    for qa_pair in qa_pairs:
        print(qa_pair)
        question = qa_pair["question"]
        response, cost = get_qa_response(model_type, agents, question, 
                                         config_list, autogen_logs_filename)
        costs.append(float(cost))
        predictions.append({"response": response})
    cost = round(float(sum(costs)), 5)

    # 3. Evaluate and calculate performance metric
    graded_outputs = run_evaluation_chain(qa_pairs, predictions)
    score = float(get_precision_score(graded_outputs))

    # 4. Reformat and evaluate using squad
    modified_qa_pairs = modify_qa_pairs(qa_pairs, predictions)    
    squad_score = get_squad_score(modified_qa_pairs, predictions)
    exact_match = squad_score['exact_match']
    f1 = squad_score['f1']

    return score, cost, exact_match, f1


def main(qa_pairs: list, config: dict, sweep_id: str):
    """Main function to initialize wandb and log the results."""
    wandb.require("service")
    with wandb.init(config=config):
        pp.pprint(wandb.config)
        config = wandb.config
        autogen_logs_filename = f"./autogen_logs/logs_filename_{sweep_id}"

        # Get the score from the objective function
        score, cost, exact_match, f1 = objective(config, qa_pairs, autogen_logs_filename)

        wandb.log({
            "cost": cost,
            "exact_match": exact_match,
            "precision": score, 
            "f1": f1
        })
        wandb.finish()
    wandb.finish()

if __name__ == "__main__":
    sweep_configuration = load_sweep_config()
    chunk_size = 1000 # Size of each document in the pdf
    num_qa_pairs = 15 # Number of question/answers to generate
    n_wandb_agents = 10 # Number of runs 

    print(f"chunk_size: {chunk_size}")
    print(f"num_qa_pairs: {num_qa_pairs}")

    # 0. Create an evaluation dataset
    print('Generating eval dataset...')
    qa_pairs = generate_eval_dataset(
                            doc_file=DOC_FILE, 
                            chunk_size=1000, 
                            num_qa_pairs = num_qa_pairs, 
                            save_to_file=True
                        )

    print('Initializing W&B Sweep...')
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)
    main_func = functools.partial(main, qa_pairs, sweep_configuration, sweep_id)
    wandb.agent(sweep_id, function=main_func, count=n_wandb_agents)
    
    wandb.finish()

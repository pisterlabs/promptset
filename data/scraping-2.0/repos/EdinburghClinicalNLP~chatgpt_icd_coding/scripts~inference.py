import argparse
import os
import sys

sys.path.append(os.getcwd())

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv("env/.env")

from src import utils
from src.api import ChatGPTCall
from src.configs import Configs


def parse_configs() -> Configs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--existing_output_dir", type=str)
    args = parser.parse_args()
    configs = Configs(**utils.load_yaml(args.config_filepath))

    return configs, args


def main():
    configs, args = parse_configs()

    utils.setup_random_seed(configs.training_configs.random_seed)

    if not args.existing_output_dir:
        # Setup experiment folder to store config file used for the API call and the predictions
        outputs_dir = utils.setup_experiment_folder(
            os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
        )
        utils.save_training_configs(configs, outputs_dir)
    else:
        outputs_dir = args.existing_output_dir

    # Setup OpenAI API configs
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_version = configs.api_configs.api_version

    dataset = pd.read_csv(configs.training_configs.dataset_path)
    print(f"Predicting {len(dataset)} test clinical notes")

    chatgpt_api = ChatGPTCall(configs, outputs_dir)

    chatgpt_outputs = chatgpt_api.predict(dataset)

    print("Prediction finished!")


if __name__ == "__main__":
    main()

import os
import sys
import json
import math
import openai
import logging
import transformers
import pandas as pd


from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    pipeline,
    set_seed,
)

from src.config import CONFIG
from src.e2e.e2e_utils import E2E_InstrucTOD, get_subset_multi
from src.data import MWOZ_Dataset
from src.utils.args_helper import DataArguments, ModelArguments, PromptingArguments

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, PromptingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, prompting_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, prompting_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = prompting_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    mwoz = MWOZ_Dataset(CONFIG, data_args)
    dataset = mwoz.dataset
    
    print(f"Task - {prompting_args.task}")
    print(f"Saving at - {data_args.save_path}")

    openai.organization = CONFIG["openai_organization"]
    openai.api_key= CONFIG["openai_api_key"]

    def completion(model_args, prompt):
        if "gpt-3.5-turbo" in model_args.model_name_or_path or "gpt-4" in model_args.model_name_or_path:
            completion = openai.ChatCompletion.create(
                model=model_args.model_name_or_path.replace("openai/", ""),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            response = completion.choices[0].message.content.strip()
        else:
            completion = openai.Completion.create(
                model=model_args.model_name_or_path.replace("openai/", ""),
                prompt=prompt,
            )
            response = completion.choices[0].text.strip()
        return response
    
    if data_args.debug_mode:
        dataset = dataset[0:9]
    else:
        dataset = dataset[data_args.start_idx:]
        
    if data_args.multi_only:
        dataset = get_subset_multi(dataset)
        
    if prompting_args.task == "e2e_instructod":
        instructod = E2E_InstrucTOD(CONFIG,
                                    model_args,
                                    data_args,
                                    dataset)
        if data_args.do_inference:
            preds = instructod.inference()
        
        
    
    else:
        outputs = dataset.copy(deep=True)
        preds = []
        prompts = []
        gold_responses = []
        idxs = []
        for idx, row in tqdm(dataset.iterrows()):

            if prompting_args.task == "rg":
                prompt = row["prompt_rg"]
                gold_response = row["gold_response"]

            elif prompting_args.task == "e2e":
                prompt = row["prompt_e2e"]
                gold_response = row["gold_response"]
                
            elif prompting_args.task == "dst":
                prompt = row["prompt_dst"]
                gold_response = row["gold_bs"]

            sample_id = row["id"]
            retry_count = 0
            while True:
                if retry_count > 5:
                    print("Retried too many times")
                    break
                try:
                    pred = completion(model_args, prompt)
                    break
                except:
                    retry_count += 1
            if retry_count > 5:
                break

            preds.append(pred)
            gold_responses.append(gold_response)
            idxs.append(sample_id)
            prompts.append(prompt)

            if idx % data_args.save_every == 0:
                temp_save_path = data_args.save_path[:-4] + "_latestSave.csv"
                temp_df = pd.DataFrame({"id":idxs,
                                        "prompts":prompts,
                                        "gold_response":gold_responses,
                                        "preds":preds})
                temp_df.to_csv(temp_save_path)

        outputs["preds"] = preds
        df = pd.DataFrame(outputs)
        df.to_csv(data_args.save_path)
        
        
if __name__ == "__main__":
    main()
    
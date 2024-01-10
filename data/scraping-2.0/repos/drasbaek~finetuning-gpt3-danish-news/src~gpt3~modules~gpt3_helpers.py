'''
Helper functions for finetuning and generating text with GPT-3

Written for the paper titled "Fine-tuning GPT-3 for Synthetic Danish News Generation" (Almasi & Schiønning, 2023).
'''

# utils 
import pathlib
import json 
import re

# gpt3 finetuning
import openai 

def import_token(token_path:pathlib.Path):
    '''
    Import OPENAI token from token.txt in specified path.

    Args
        token_path: path of token.txt file 
    '''
    # get token from txt
    with open(token_path) as f:
        openai.api_key = f.read()

def remove_datetime_from_model(input_string):
    """
    Remove date and time from the input string.

    Args:
        input_string (str): The input string.

    Returns:
        str: The input string with date and time removed.
    """
    model_parts = input_string.split(":")
    last_part = model_parts[-1]
    if "-" in last_part:
        last_part_without_datetime = '-'.join(last_part.split("-")[:-6])
    else:
        last_part_without_datetime = last_part.split(":")[0]
    return ':'.join(model_parts[:-1]) + ":" + last_part_without_datetime

def find_existing_finetune_id(existing_finetunes, target_finetune):
    """
    Find the ID of an existing fine-tune with the specified target_finetune suffix.

    Args:
        existing_finetunes (dict): A dictionary containing the list of existing fine-tunes.
        target_finetune (str): The name to identify the specific fine-tune.

    Returns:
        str or None: The ID of the existing fine-tune, or None if not found.
    """
    existing_finetune_id = None
    for finetune_info in existing_finetunes["data"]:
        try:
            finetuned_mdl = finetune_info.get("fine_tuned_model")
            if finetuned_mdl: # only if there is a finetuned_mdl (if it is cancelled, it may be recorded as existing, but with no entry in "fine_tuned_model"
                # remove datetime from mdl suffix
                finetuned_mdl_without_dt = remove_datetime_from_model(finetuned_mdl)

                # remove everything but the target name (specified mdl suffix)
                finetuned_name = finetuned_mdl_without_dt.split(":")[-1]

                # if the target_finetune is the same as the mdl suffix in existing_finetunes, then update the existing_finetune_id
                if target_finetune == finetuned_name:
                    existing_finetune_id = finetune_info.get("id")
                    fine_tuned_model = finetune_info.get("fine_tuned_model")  
                    break
        except KeyError:
            print("Error: finetune_info has no 'fine_tuned_model' key.")

    return existing_finetune_id, fine_tuned_model


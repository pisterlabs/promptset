import datetime
import argparse
import yaml
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.file_object import FileObject
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.types.fine_tuning.job_create_params import Hyperparameters
from utils import (
    OBJECT_FINE_TUNE,
)


def upload_dataset(args):
    print("Uploading dataset to OpenAI.")
    if not is_file_uploaded(args):
        openai_file_info = args.openai_client.files.create(
            file=open(args.training_file, 'rb'),
            purpose='fine-tune'
        )
        args.openai_file_info = openai_file_info
        
        
def fine_tune(args):
    print("Start fine-tuning.")
    openai_ft_job_info = args.openai_client.fine_tuning.jobs.create(
        training_file = args.openai_file_info.id,
        model = args.model_id,
        hyperparameters = {
            "n_epochs": args.n_epochs,
        }
    )

    args.openai_ft_job_info = openai_ft_job_info


def is_file_uploaded(args):
    list_files = args.openai_client.files.list()
    if not list_files:
        return False
    
    for f in list_files:
        if f.filename == os.path.basename(args.training_file):
            print(f"File {f.filename} is already uploaded.")
            if args.delete_old_dataset:
                print(f"Deleting old dataset {f.filename}.")
                args.openai_client.files.delete(f.id)
                continue
            else:
                args.openai_file_info = f
                return True
    return False

def serialize_hyperparameters(obj):
    return {'n_epochs': obj.n_epochs, "batch_size": obj.batch_size, "learning_rate_multiplier": obj.learning_rate_multiplier}

def write_info(info_object, output_dir):
    path_file = os.path.join(output_dir, info_object.id + ".json")
    dict_info = vars(info_object)

    if info_object.object == OBJECT_FINE_TUNE:
        dict_info['hyperparameters'] = serialize_hyperparameters(dict_info['hyperparameters'])
        #print(dict_info['hyperparameters'])

    with open(path_file, 'w') as f_writer:
        json.dump(dict_info, f_writer)

def main(args):
    # upload dataset to openai
    upload_dataset(args)

    # write openai file info
    write_info(args.openai_file_info, args.output_dir)
    print(f"Dataset info is saved to directory: {args.output_dir}")

    # fine tune
    fine_tune(args)

    # write openai fine tune job info
    write_info(args.openai_ft_job_info, args.output_dir)
    print(f"Fine-tuned model is saved to directory: {args.output_dir}")



if __name__ == '__main__':
    load_dotenv()
    if os.getenv('OPENAI_API_KEY', None) is None:
        print("OPENAI_API_KEY is not set.")
        sys.exit(1)
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    parser = argparse.ArgumentParser("Fine-tune GPT-3 on a dataset.")
    parser.add_argument("--config_file", type=str, default="configure/openai.yaml", help="The model ID to use for fine-tuning.")
    parser.add_argument("--training_file", type=str, default="data/processed/nyt-2020-openai.jsonl", help="The path to the data directory.")
    parser.add_argument("--openai_client", type=OpenAI, default=client, help="openai client")
    parser.add_argument("--openai_file_info", type=FileObject, help="openai train file info")
    parser.add_argument("--openai_ft_job_info", type=FineTuningJob, help="openai fine-tune job info")
    parser.add_argument("--output_dir", type=str, default="models", help="The path to save fine-tuned model.")
    parser.add_argument("--delete_old_dataset", action='store_true', help="Delete old dataset.")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, 'r'))
    args.__dict__.update(config)   
    
    main(args)

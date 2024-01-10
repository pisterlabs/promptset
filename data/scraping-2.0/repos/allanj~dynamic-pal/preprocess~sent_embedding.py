
from pal.utils import read_data
from tqdm import tqdm
import openai
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from simcse import SimCSE
import argparse
"""
Prompting the sentence embeddings
"""

name2model = {
    "sentence-transformers": SentenceTransformer,
    "princeton-nlp": SimCSE
}

def parse_arguments(parser):
    parser.add_argument('--openai_key', default="", type=str)
    parser.add_argument('--dataset', default='gsm8k', type=str, choices=["gsm8k", "svamp", "MathQA"])
    parser.add_argument('--embedding_model_name', default='text-embedding-ada-002', type=str)
    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(f"{k} = {args.__dict__[k]}",flush=True)
    return args

def get_embedding(text, model="text-embedding-ada-002"):
    return np.array(openai.Embedding.create(input=[text], model=model)['data'][0]['embedding'])

def prompt_emb(input_file, output_file, dataset_name,
               model_name= "text-embedding-ada-002",
               local_model = None):
    data = read_data(input_file)
    all_reps = []
    for i in tqdm(range(len(data)), desc="Processing", total=len(data)):
        obj = data[i]
        if dataset_name == "gsm8k":
            question = obj['question']
        elif dataset_name == "svamp":
            question = obj['question'] if 'question' in obj else obj['sQuestion'].strip() ## for svamp
        elif dataset_name == "MathQA":
            question = obj["Problem"].strip()
        else:
            raise NotImplementedError
        success = False
        count = 1
        if model_name == "text-embedding-ada-002":
            while not success:
                try:
                    current_rep = get_embedding(question, model=model_name)
                    success = True
                except:
                    print(f"Not success, sleeping for {count} then retrying", flush=True)
                    time.sleep(count)
                    count *= 2
                    continue
        elif model_name.startswith("sentence-transformers"):
            current_rep = local_model.encode([question])[0]
        elif model_name.startswith("princeton-nlp"):
            ## simcse
            current_rep = local_model.encode(question).tolist()
        all_reps.append(current_rep)
        time.sleep(1)
        if len(all_reps) % 50 == 0:
            res = np.stack(all_reps)
            np.save(output_file, res)
    res = np.stack(all_reps)
    np.save(output_file, res)

def load_sent_model(model_name: str = 'sentence-transformers/sentence-t5-base', device: str = 'cuda'):
    model = name2model[model_name.split("/")[0]](model_name, device=device)
    return model


if __name__ == '__main__':
    args = parse_arguments(argparse.ArgumentParser())
    dataset = args.dataset
    openai.api_key = args.openai_key  ##TODO: add your own key
    embedding_model_name = args.embedding_model_name
    local_model = None
    if embedding_model_name != "text-embedding-ada-002":
        local_model = load_sent_model(model_name=embedding_model_name)
    suffix = embedding_model_name.split("/")[-1]
    if dataset == 'gsm8k':
        prompt_emb(input_file="datasets/gsm8k/gsm8k_train_sent_split.json",
                   output_file=f"datasets/gsm8k/gsm8k_train_sent_emb_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
        prompt_emb(input_file="datasets/gsm8k/gsm8k_test_sent_split.json",
                   output_file=f"datasets/gsm8k/gsm8k_test_sent_emb_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
    elif dataset == "svamp":
        prompt_emb(input_file="datasets/svamp/trainset_nodup.json",
                   output_file=f"datasets/svamp/trainset_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
        prompt_emb(input_file="datasets/svamp/testset_nodup.json",
                   output_file=f"datasets/svamp/testset_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
    elif dataset == "MathQA":
        prompt_emb(input_file="datasets/MathQA/mathqa_train_nodup_our_filtered.json",
                   output_file=f"datasets/MathQA/mathqa_train_emb_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
        prompt_emb(input_file="datasets/MathQA/mathqa_test_nodup_our_filtered.json",
                   output_file=f"datasets/MathQA/mathqa_test_emb_{suffix}.npy", dataset_name=dataset,
                   model_name= embedding_model_name, local_model=local_model)
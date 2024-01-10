__author__ = "Jon Ball"
__version__ = "November 2023"

from notebooks.gpt_utils import (start_chat, user_turn, system_turn)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import chromadb
import jinja2
import torch
import argparse
import random
import json
import time
import os


def main(args):
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # Document embedding model
    model_name = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    # Access local chroma client
    persistent_client = chromadb.PersistentClient(path="chroma")
    # Vector db of 10 hand-labeled examples
    manualDB = Chroma(
        client=persistent_client,
        collection_name="manual",
        embedding_function=embedding,
        )
    # Vector db of 90 examples labeled by GPT-4
    gpt4DB = Chroma(
        client=persistent_client,
        collection_name="gpt4",
        embedding_function=embedding
        )
    #
    jinjitsu = jinjaLoader(args.template_dir, args.template_file)
    #
    # load records
    with open(args.input_file, "r") as infile:
        resp = json.load(infile)
    records = resp["response"]["docs"]
    # create the output dir if it doesn't exist
    if not os.path.exists(args.completion_dir):
        os.makedirs(args.completion_dir)
    # loop over records
    for rec in tqdm(records):
        time.sleep(random.randint(1, 3))
        input = str(rec)
        # Pull most similar manually labeled doc
        examples = [ex.metadata for ex in manualDB.similarity_search(input, 1)]
        if args.n_examples > 1:
            # Pull most similar GPT-4 labeled docs
            examples += [ex.metadata for ex in gpt4DB.similarity_search(input, args.n_examples-1)]
        templateVars = {
            "examples": examples,
            "input": input, 
            "output": ""
            }
        PROMPT = jinjitsu.render(templateVars)
        # Get ChatGPT compeletion
        chat = start_chat("You are a sociological research assistant tasked with labeling records of articles published in Sociology of Education. You return only correct labels in the specified format.")
        chat = user_turn(chat, PROMPT)
        try:
            chat = system_turn(chat, model=args.model_name)
        except TimeoutError:
            # try again
            time.sleep(random.randint(1, 3))
            chat = system_turn(chat, model=args.model_name)
        # Save json
        templateVars["output"] = chat[-1]["content"]
        with open(os.path.join(args.completion_dir, rec["id"] + ".json"), "w") as outfile:
            json.dump(templateVars, outfile)
    

class jinjaLoader():
    # jinja2 template renderer
    def __init__(self, template_dir, template_file):
        self.templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.templateEnv = jinja2.Environment( loader=self.templateLoader )
        self.template = self.templateEnv.get_template( template_file )

    def render(self, templateVars):
        return self.template.render( templateVars )


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--input_file", type=str, default="data/eric/se.json")
    parser.add_argument("--completion_dir", type=str, default="completions/SE/chatgpt/fiveshot")
    parser.add_argument("--template_dir", type=str, default="prompts")
    parser.add_argument("--template_file", type=str, default="fewshot.prompt")
    parser.add_argument("--n_examples", type=int, default=5)
    args = parser.parse_args()
    # run
    main(args)

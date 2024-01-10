__author__ = "Jon Ball"
__version__ = "December 2023"

from openai_utils import (start_chat, user_turn, system_turn)
from tqdm import tqdm
import torch
import random
import time
import os

def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # loop over prompts
    if not os.path.exists("completions"):
        os.mkdir("completions")
    # HF model used to retrieve prompts
    HF_MODEL = "Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit"
    # OpenAI model used to retrieve prompts
    OPENAI_MODEL = "text-embedding-ada-002"
    print(f"GPT-4 generating completions for prompts with checklists retrieved by {HF_MODEL}...")
    walk_articles("prompts/2_7B", "completions/HF")
    print("   ...completions generated.")
    print(f"GPT-4 generating completions for prompts with checklists retrieved by {OPENAI_MODEL}...")
    walk_articles("prompts/openai", "completions/openai")
    print("   ...completions generated.")
    print("Done.")


def walk_articles(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for dirname, dirpath, filenames in os.walk(input_dir):
        for filename in tqdm([f for f in filenames if f.endswith(".prompt")]):
            with open(os.path.join(dirname, filename), "r") as infile:
                prompt = infile.read()
            completion = get_completion(prompt)
            with open(os.path.join(output_dir, filename[:-7] + ".txt"), "w") as outfile:
                outfile.write(completion)
                

def get_completion(prompt, model="gpt-4-1106-preview"):
    chat = start_chat("You are an expert biostatistician, methodologist, and reviewer of research articles. You are reviewing a biomedical or health research article, according to a research reporting guideline checklist. RETURN A COMPLETED VERSION OF THE CHECKLIST WITH CORRECT ANSWERS.")
    chat = user_turn(chat, prompt)
    try:
        chat = system_turn(chat, model=model)
    except TimeoutError:
        # try again
        time.sleep(random.randint(1, 3))
        chat = system_turn(chat, model=model)
    return chat[-1]["content"]


if __name__ == "__main__":
    main()
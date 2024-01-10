
import argparse
import time
import json

from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class NewTokenHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens_generated = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running."""
        self.num_tokens_generated = 0
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        total_time = time.time() - self.start_time
        print(f"\n\n {self.num_tokens_generated} tokens generated in {total_time:.2f} seconds")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.num_tokens_generated += 1
        print(f"{token}", end="", flush=True)

def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def run_on_user_input(chain, args):
    # Just run a command prompt
    while True:
        prompt = input('> ')

        chain.run(joke=prompt)
        print("\n")

def run_on_examples(chain, args):
    # Run on examples from a file
    prompts = []
    for line in open(args.input_file, 'r'):
        data = json.loads(line)
        prompts.append(data)

    out = None
    if args.output_file != '':
        out = open(args.output_file, 'w')

    total_processed = 0
    total_was_funny = 0
    for data in prompts:
        prompt = data['prompt']
        completion = data['completion'] # a good completion

        print(prompt)
        model_completion = chain.run(joke=prompt)
        print("\n")

        if out is not None:
            was_funny = input("Was that funny? (y/n) > ")

            if 'q' == was_funny.lower():
                exit()

            was_funny = was_funny.lower() == 'y'

            total_processed += 1
            if was_funny:
                total_was_funny += 1

            accuracy = total_was_funny / total_processed * 100
            print(f"Funny Meter {total_was_funny}/{total_processed} {accuracy:.2f}%\n\n")

            out_data = {
                'prompt': prompt,
                'model_completion': model_completion,
                'completion': completion,
                'was_funny': was_funny
            }
            out.write(json.dumps(out_data) + "\n")
            out.flush()

def main(args):
    # Local CTransformers wrapper for Llama-2-7B-Chat
    llm = CTransformers(
        model=args.model, # Location of downloaded GGML model
        model_type=args.model_type, # Model type Llama
        stream=True,
        callbacks=[NewTokenHandler()],
        config={
            'max_new_tokens': 256,
            'temperature': args.temperature,
            'stop': "<0x0A>"
        }
    )

    prompt_str = read_file(args.prompt)
    prompt = PromptTemplate.from_template(prompt_str)
    chain = LLMChain(llm=llm, prompt=prompt)

    run_on_examples(chain, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/llama-2-7b-chat.ggmlv3.q8_0.bin')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
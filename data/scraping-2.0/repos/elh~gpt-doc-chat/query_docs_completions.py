import sys
import os
import datetime
import argparse
from dotenv import load_dotenv
import openai
import pandas as pd
import numpy as np

from semantic_search import semantic_search

'''
WARN: powered by GPT completions. Prefer query_docs
'''

MODEL = "text-davinci-003"
LOGS_DIRECTORY = "logs/query_docs_completions"

BASE_PROMPT = "Given the following DOCUMENTATION please answer the following QUESTION."

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description='Answer questions based on a corpus of documents')
    parser.add_argument('--embedding_csv', type=str, default="", help='embedding csv file')
    parser.add_argument('--question', type=str, default="", help='your question about the docs')
    parser.add_argument('--prompt', type=str, default="", help='Customized prompt to be prepended to base system prompt (optional)')
    args = parser.parse_args()
    if args.question == "" or args.embedding_csv == "":
        print("ERROR: Embedding CSV and question are required.")
        sys.exit(1)

    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    now = datetime.datetime.now()
    log_file_name = LOGS_DIRECTORY + "/" + str(now.strftime("%Y_%m_%d_%H.%M.%S")) + ".txt"

    ################ Semantic search for most similar documents ################
    df = pd.read_csv(args.embedding_csv)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    results = semantic_search(df, args.question, n=5)
    print(results)

    ############################## Build up prompt ##############################
    prompt = args.prompt + "." + BASE_PROMPT + "\n\nDOCUMENTATION:\n"

    token_sum = 0
    for i in results.index:
        content = results['content'][i]
        total_tokens = results['total_tokens'][i]

        # bail if over token limit
        token_sum += total_tokens
        if token_sum > 3072:
            break
        prompt += content

    prompt += "\n\nQUESTION:\n" + args.question + "\n\nANSWER:"
    print(prompt)

    ######################### Hit GPT for completion ##########################
    resp = openai.Completion.create(model=MODEL, prompt=prompt, max_tokens=512)
    output = resp.get("choices", [{}])[0].get("text", "").lstrip('\n').lstrip(' ')
    if output == "":
        print("ERROR: No response from OpenAI 🤖\n" + resp)
        sys.exit(1)
    total_tokens = resp.get("usage", {}).get("total_tokens", 0)

    print(output)
    print("\n----------\n" + str(total_tokens) + " tokens. model: " + MODEL)

    open(log_file_name, 'w').write(prompt + "\n" + output + "\n\n----------\n" + str(total_tokens) + " tokens. model: " + MODEL) # full reset of file

if __name__ == "__main__":
    main()

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
NOTE: powered by GPT chat
'''

MODEL = "gpt-3.5-turbo"
LOGS_DIRECTORY = "logs/query_docs"

BASE_PROMPT = "Please answer the question based on the DOCUMENTATION. Please say 'I don't know' if you don't know the answer.'"

def query_docs(embedding_csv, question, prompt=""):
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    now = datetime.datetime.now()
    log_file_name = LOGS_DIRECTORY + "/" + str(now.strftime("%Y_%m_%d_%H.%M.%S")) + ".txt"

    ################ Semantic search for most similar documents ################
    df = pd.read_csv(embedding_csv)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    results = semantic_search(df, question, n=5)
    print(results)

    ############################## Build up prompt ##############################
    system_prompt = prompt + "." + BASE_PROMPT + "\n\nDOCUMENTATION:\n"

    token_sum = 0
    for i in results.index:
        content = results['content'][i]
        total_tokens = results['total_tokens'][i]

        # bail if over token limit
        token_sum += total_tokens
        if token_sum > 3072:
            break
        system_prompt += content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    print("SYSTEM PROMPT:\n" +
          system_prompt + "\n\n" +
          "QUESTION:\n" +
          question + "\n\n")

    ######################### Hit GPT for completion ##########################
    resp = openai.ChatCompletion.create(model=MODEL, messages=messages, max_tokens=512)
    output = resp.get("choices", [{}])[0].get("message", {}).get("content", "").lstrip('\n').lstrip(' ')
    if output == "":
        print("ERROR: No response from OpenAI 🤖\n" + resp)
        sys.exit(1)
    total_tokens = resp.get("usage", {}).get("total_tokens", 0)

    print("REPLY:\n" +
          output + "\n\n----------\n" +
          str(total_tokens) + " tokens. model: " + MODEL)

    open(log_file_name, 'w').write("SYSTEM PROMPT:\n" +
                                   system_prompt + "\n\n" +
                                   "QUESTION:\n" +
                                   question + "\n\n" +
                                   "REPLY:\n" +
                                   output + "\n\n----------\n" +
                                   str(total_tokens) + " tokens. model: " + MODEL) # full reset of file

    return output

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

    query_docs(args.embedding_csv, args.question, args.prompt)

if __name__ == "__main__":
    main()

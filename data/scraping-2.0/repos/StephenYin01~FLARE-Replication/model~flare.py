import os
import json
import math
import argparse

from openai_api import QueryAgent

if __name__ == "__main__":

    cur_path = os.path.abspath(os.curdir)

    # Read the arguments
    parser = argparse.ArgumentParser(description="Define experiment")
    parser.add_argument("-d", "--dataset", type=str, default="ASQA", help="Name of ASQA dev dataset")
    parser.add_argument("-n", "--name", type=str, default="final-predictions", help="Name of the experiment")
    # Parse the arguments
    args = parser.parse_args()

    # Retrieve the API key saved in environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Instatiate the Query Agent for OpenAI API Calls
    with open(cur_path + "/configs/asqa.json", 'r') as f:

        # We use gpt-3.5-turbo-instruct for cost reduction (Input: $0.0015 / 1K tokens  Output: $0.0020 / 1K tokens) instead of ($0.0200 / 1K tokens)
        qa = QueryAgent(
            model='gpt-3.5-turbo-instruct',
            retrieval_kwargs=json.load(f),
            api_key=api_key, 
        )

    # Retrieve the evaluation questions
    with open(cur_path + f"/dataset/{args.dataset}.json", 'r') as f:

        questions = json.load(f)
        questions = questions['dev']

    # Gather predictions
    num_qs = len(questions)
    batch_size = 20
    batch_num = 0
    count = 0
    predictions = dict()

    batch_keys = []
    batch_questions = []
    batch_responses = []
    
    for k, v in questions.items():

        batch_keys.append(k)
        batch_questions.append(v['ambiguous_question'])

        count += 1

        if count % batch_size == 0 or count == num_qs:
            # Track batch num
            batch_num += 1
            bs = count - batch_size * (batch_num - 1)
            print(f"Batch {batch_num} / {math.ceil(num_qs/batch_size)}, Size: {bs}")

            # Query the agent
            batch_responses = qa.respond(batch_questions)

            # Save the results
            for i in range(bs):
                predictions[batch_keys[i]] = batch_responses[i]

            # Reset batch variables
            batch_keys = []
            batch_questions = []
            batch_responses = []

    with open(cur_path + f"/outputs/{args.name}.json", 'w') as f:

        json.dump(predictions, f)

    # Save analytics
    qa._save_analytics(f"/outputs/{args.name}-analytics.json")
    
import os
import torch
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from src.cf_generation.llm_generation.utils import save_to_disk

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def get_response(prompt, max_tokens, model_name):
    """
    :param prompt:
    :param max_tokens:
    :param model_name:
    :return:

    Sample output
    generated_ans =
    {
          "id": "chatcmpl-7dKbsNzkrAjNumSBMc4IhwCZAEMlo",
          "object": "chat.completion",
          "created": 1689608372,
          "model": "gpt-4-0613",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "The phenomena of electric and magnetic fields being perpendicular to each other is a
                fundamental part of how electromagnetic waves work. \n\nLet's consider an analogy to understand
                this better. ...},
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 222,
            "completion_tokens": 247,
            "total_tokens": 469
          }
    }

    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]

    model_name = "gpt-4-0314"
    save_path = (
        BASE_PATH + f"src/data/squad/{model_name}_qa_relevance_seed_{args.seed}/"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(
        BASE_PATH,
        f"src/data/squad/counterfactual_samples_flan_t5_xxl_context_irrelevance.jsonl",
    )
    files = [file_path]
    skipped = 0
    c = 0
    for file_name in files:
        examples = []
        with jsonlines.open(file_name) as reader:
            for example in tqdm(reader):
                try:
                    c += 1
                    # if c <= 100:
                    #     continue
                    id = example["id"].split("_")[0]
                    context = example["context"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]

                    # print("Given ans:", example["answers"])

                    orig_example = [
                        sample for sample in squad_data if sample["id"] == id
                    ][0]

                    orig_context = orig_example["context"]
                    orig_question = orig_example["question"]
                    orig_answer = orig_example["answers"]

                    # input = GRADE_DOCS_PROMPT_FAST.format(query=question, result=context, answer=answer)
                    # print(input)

                    template = (
                        f"Given the question: \n"
                        f"{question} \n"
                        f"Decide if the following retrieved context is relevant to the {answer}: \n"
                        f"{context} \n"
                        "Answer in the following format: \n"
                        "'Context is relevant: True or False.' \n"
                    )
                    # print(template)
                    # break

                    output = get_response(
                        prompt=template, max_tokens=10, model_name=model_name
                    )

                    result = {
                        "id": example["id"],
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "context_relevance": output,
                    }
                    # print(result)
                    # break
                    examples.append(result)
                    if c == 500:
                        break
                except Exception as e:
                    print("Skip")
        # # save the remaining examples
        if examples:
            save_to_disk(
                examples,
                f"{save_path}counterfactual_samples_{model_name}_flan_t5_xxl_{c}.jsonl",
            )
        # print(examples)

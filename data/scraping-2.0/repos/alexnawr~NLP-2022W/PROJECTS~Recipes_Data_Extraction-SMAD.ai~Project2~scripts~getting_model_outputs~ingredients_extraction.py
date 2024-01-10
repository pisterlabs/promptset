import json
import time
from argparse import ArgumentParser

import openai
import pandas as pd
from tqdm import tqdm

MODEL_TO_USE = "text-davinci-003"


def get_response(text):
    """Sends prompt to OpenAI API and returns response

    Args:
        text (str): prompt which is passed to model

    Returns:
        dict: response dict.
    """
    model_input = ner_prompt.replace("INPUT_TEXT", text)
    response = openai.Completion.create(
        model=MODEL_TO_USE,
        prompt=model_input,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response


def main(args):
    # Dataloading
    data = pd.read_csv("dataset/llm_sampled_data.csv")
    # Setting API key
    with open("openai_key.txt", encoding="utf-8") as f:
        key = f.readline()
    openai.api_key = key
    # Loading prompt template
    with open("prompts_templates.json", encoding="utf-8") as f:
        ner_prompt = json.load(f)["ingredients_extraction"][args.prompt_number]
    # Creating list of prompts to be sent to API
    inputs = [
        ner_prompt.replace("INPUT_TEXT", text)
        for text in data.ingredients[: args.number_examples]
    ]

    responses = []
    # Getting resposnes for all inputs
    for text in tqdm(inputs):
        response = openai.Completion.create(
            model=MODEL_TO_USE,
            prompt=text,
            temperature=0,
            max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        responses += [json.loads(x["text"]) for x in response["choices"]]
        # Adding sleep because of API requests limits
        time.sleep(2)

    # Creating results dataframe and saving it
    results_dataframe = pd.DataFrame(
        {
            "ingredients_list": data.ingredients[: len(responses)].tolist(),
            "gpt-3_extracted": responses,
        }
    )

    results_dataframe.to_csv("results/ingredients_extraction_results.csv", index=False)


if __name__ == "__main__":
    # Adding argument parser.
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--number_examples",
        default=50,
        type=int,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "-p",
        "--prompt_number",
        default=1,
        type=int,
        help="Prompt number to use",
    )
    args = parser.parse_args()
    main(args)

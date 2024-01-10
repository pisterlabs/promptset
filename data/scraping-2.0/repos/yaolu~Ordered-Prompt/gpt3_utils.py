import os

import openai
import pickle
from tqdm import tqdm
import sys
import argparse

import debugger

def inference_gpt3_prediction(prompt, engine="ada"):
    """
    We only implement the complete one token version
    :param prompt: str
    :param engine: str, choose from ada, babbage, curie, davinci
    :return:
    """
    openai.organization = os.environ["openai_organization"]
    openai.api_key = os.environ["openai_api_key"]

    assert type(prompt) == str

    response = openai.Completion.create(engine=engine,
                                        prompt=f"{prompt}",
                                        temperature=0,
                                        max_tokens=1,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        stop=["\n"],
                                        logprobs=100)
    prediction_dist = response["choices"][0]['logprobs']['top_logprobs'][0]

    return response, prediction_dist


def inference_gpt3_compeletion(prompt_text, max_tokens=128, temperature=2.0, top_p=0.1,
                               presence_penalty=2.0, engine="ada"):
    """
    We only implement the complete one token version
    :param prompt: str
    :param engine: str, choose from ada, babbage, curie, davinci
    :return:
    """
    openai.organization = os.environ["openai_organization"]
    openai.api_key = os.environ["openai_api_key"]

    assert type(prompt_text) == str

    response = openai.Completion.create(engine=engine,
                                        prompt=f"{prompt_text}",
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        top_p=top_p,
                                        presence_penalty=presence_penalty,
                                        logprobs=100)

    compeletion_text = response['choices'][0]['text']
    return response, compeletion_text

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--engine", type=str, default="ada")
#     parser.add_argument("--ckpt", type=str, required=True)
#     args = parser.parse_args()
#
#     engine = args.engine
#     ckpt = pickle.load(open(args.ckpt, "rb"))
#     result = []
#
#     target_labels = set(ckpt["config"]["label_mapping"].values())
#
#     for sentence in tqdm(ckpt["result"]):
#         prompts = sentence["prompt_sequence_text"]
#
#         prediction = {"token": [], "dist": []}
#         _token = []
#         _dist = []
#         for prompt in prompts:
#             response = inference_gpt3(prompt[0], target_labels=target_labels, engine=engine)
#             _token.append(response["prediction_token"])
#             _dist.append(response["prediction_dist"])
#         prediction["prediction_text"] = " ".join(_token)
#         prediction["prediction_dist"] = _dist
#         prediction["label"] = sentence["label"]
#         result.append(prediction)
#
#     config = ckpt["config"]
#     config["model"] = f"gpt3-{engine}"
#
#     with open(f"gpt3_{engine}_{args.ckpt}", 'wb') as fout:
#         pickle.dump({"result": result, "config": config}, fout)
#

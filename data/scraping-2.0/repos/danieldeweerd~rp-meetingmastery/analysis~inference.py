import os

import openai
import replicate
import requests
from dotenv import load_dotenv

models = ["gpt-2", "gpt-3.5-turbo", "flan-t5-xxl"]
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")

urls = {
    "flan-t5-small": "replicate/flan-t5-small:69716ad8c34274043bf4a135b7315c7c569ec931d8f23d6826e249e1c142a264",
    "flan-t5-xxl": "replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
    "flan-t5-base": "https://api-inference.huggingface.co/models/google/flan-t5-base",
    "gpt-2": "https://api-inference.huggingface.co/models/gpt2",
    "vicuna-13b": "replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e",
    "oasst-pythia-12b": "replicate/oasst-sft-1-pythia-12b:28d1875590308642710f46489b97632c0df55adb5078d43064e1dc0bf68117c3",
    "stablelm-7b": "stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb",
    "dolly-12b": "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    "dolly-gptj": "cjwbw/dolly:fe699f6290c55cb6bac0f023f5d88a8faba35e2761954e4e0fa030e2fdecafea",
    "helloworld-0b": "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa",
    "alpaca-lora-7b": "https://api-inference.huggingface.co/models/chainyo/alpaca-lora-7b"
}

openai_chat_completions = ["gpt-3.5-turbo"]
openai_completions = ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"]
replicate_models = ["vicuna-13b", "flan-t5-small", "flan-t5-xxl", "oasst-pythia-12b", "stablelm-7b", "dolly-12b",
                    "dolly-gptj"]

headers = {"Authorization": "Bearer {}".format(huggingface_api_key)}


def query(prompt, model_name="gpt-3.5-turbo", debug=False, max_length=1000):
    """
    Utility function to query models from the OpenAI and HuggingFace APIs.
    :param prompt: The prompt to query the model with.
    :param model_name: The name of the model to query.
    :return: The response from the model.
    """

    if type(prompt) == str and model_name not in replicate_models:
        prompt = [prompt]

    if model_name in openai_chat_completions:
        messages = [{"role": "user", "content": p} for p in prompt]
        response = openai.ChatCompletion.create(model=model_name, messages=messages)

        if debug:
            print(response)

        return response['choices'][0]['message']['content']

    elif model_name in openai_completions:
        response = openai.Completion.create(model=model_name, prompt=prompt, max_tokens=max_length)

        if debug:
            print(response)

        return response['choices'][0]['text']

    elif model_name in replicate_models:
        response = ""
        iterator = replicate.run(
            urls[model_name],
            input={"prompt": prompt, "max_length": max_length}
        )
        for item in iterator:
            response += item

        return response

    else:
        url = urls[model_name]
        options = {
            "use_cache": False,
            "wait_for_model": False
        }
        response = requests.post(url, headers=headers, json={"inputs": prompt, "options": options})

        if debug or True:
            print(response.json())

        return response.json()[0][0]['generated_text']


def cola(phrase):
    response = requests.post("https://api-inference.huggingface.co/models/textattack/roberta-base-CoLA",
                             headers=headers, json={"inputs": phrase})
    response = response.json()

    if type(response) == dict and response["error"] is not None and "Input is too long" in response["error"]:
        return False

    try:
        return response[0][0]["label"] == "LABEL_1"
    except KeyError:
        print("Exited due to KeyError. Response:")
        print(response)
        exit()


def continuous_cola(phrase):
    response = requests.post("https://api-inference.huggingface.co/models/textattack/roberta-base-CoLA",
                             headers=headers, json={"inputs": phrase})
    response = response.json()

    if type(response) == dict and response["error"] is not None and "Input is too long" in response["error"]:
        return False

    score_0 = response[0][0]
    score_1 = response[0][1]

    if score_0["label"] == "LABEL_1":
        return score_0["score"]
    else:
        return score_1["score"]

    # return response[0][0]["label"] == "LABEL_1"


def prettify(phrase):
    prompt = "Add proper capitalization and capitalization to the following phrase: {}".format(phrase)
    return query(prompt, model_name="gpt-3.5-turbo")


def classify_expression(thesis, expression, model_name="gpt-3.5-turbo", batch_size=10):
    template = """
    You are given a thesis and an expression. Output 'A' if the expression agrees with the thesis. Output 'D' if the expression disagrees with the thesis. 
    Output 'O' if the expression neither agrees nor disagrees with the thesis. Limit your output to the single character.

    Thesis: {}
    Expression: {}
    """

    prompt = template.format(thesis, expression)
    response = query(prompt, model_name=model_name)
    return response

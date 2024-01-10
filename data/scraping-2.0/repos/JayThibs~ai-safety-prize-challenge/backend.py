import json
import requests
import os
from utils import *
import openai
import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()

# # make sure you specify .env
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# # COHERE_API_KEY = os.environ["COHERE_API_KEY"]
# # GOOSEAI_API_KEY = os.environ["GOOSEAI_API_KEY"]
# # GPTJ_API_KEY = os.environ["GPTJ_API_KEY"]
# LABELSTUDIO_API_TOKEN = os.environ["LABELSTUDIO_API_TOKEN"]
# LABELSTUDIO_ENDPOINT = os.environ["LABELSTUDIO_ENDPOINT"]
# NGROK_API_ENDPOINT = os.environ["NGROK_API_ENDPOINT"]

# make sure you specify the secrets when running on streamlit cloud
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
# GOOSEAI_API_KEY = st.secrets["GOOSEAI_API_KEY"]
# GPTJ_API_KEY = st.secrets["GPTJ_API_KEY"]
LABELSTUDIO_API_TOKEN = st.secrets["LABELSTUDIO_API_TOKEN"]
LABELSTUDIO_ENDPOINT = st.secrets["LABELSTUDIO_ENDPOINT"]
NGROK_API_ENDPOINT = st.secrets["NGROK_API_ENDPOINT"]


def openai_inference_request(input_text, max_tokens=25, temperature=0.9, number_of_completions=1):

    loading = st.info(f"Running prediction request ...")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    n = number_of_completions
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stream=False,
        # stop=[".", "!", "?"],
    )
    loading.empty()
    results = []
    for i in range(n):
        st.markdown(response['choices'][0]['text'])
        results.append(str(response["choices"][0]["text"]))
    return results


def check_toxicity(completions):
    results = []
    for completion in completions:
        url = NGROK_API_ENDPOINT
        payload = {"data": [[completion]]}
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, json=payload, headers=headers)
        results.append(response.json()['data'][0])
    return results


def import_to_labelstudio(
    input_text, project_id, predicted_labels, predicted_scores, model_name=""
):

    url = f"{LABELSTUDIO_ENDPOINT}/api/projects/{project_id}/tasks/bulk/"

    auth_token = f"Token {LABELSTUDIO_API_TOKEN}"

    payload = [
        {
            "data": {"text": input_text, "meta_info": {"model_name": model_name}},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "category",
                            "to_name": "content",
                            "type": "choices",
                            "value": {"choices": predicted_labels},
                        }
                    ],
                }
            ],
        }
    ]

    print(payload)
    headers = {"Content-Type": "application/json", "Authorization": auth_token}

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)

import json
import os

from flask import Flask, request
import openai

app = Flask(__name__)


class Config:
    def __init__(self, api_key_file: str, model_description_file: str, model: str):
        with open(api_key_file, "r") as f:
            self.api_key = f.read().strip()

        with open(model_description_file, "r") as f:
            self.model_description = f.read().strip()

        self.model = model


def get_ctf_prompt(prompt_version: str) -> str:
    with open(f"config/ctf-promtps.json", "r") as file:
        prompt = json.load(file)

    versions = list(prompt.keys())

    if prompt_version not in versions:
        return prompt[versions[0]]

    return prompt[prompt_version]


def get_flag() -> str:
    # Exit with error if flag.txt does not exist
    if not os.path.exists(f"config/flag.txt"):
        print("Flag not found. Please run setup.py to generate a flag.")
        exit(1)

    with open(f"config/flag.txt", "r") as file:
        flag = file.read().strip()

    return flag


@app.route('/chat', methods=['POST'])
def ctf_chat():
    prompt: str = get_ctf_prompt(request.form['ctf_version']).format(flag=get_flag())

    try:
        response = openai.ChatCompletion.create(
            model=config.model,
            messages=[
                {"role": "system", "content": config.model_description},
                {"role": "user", "content": f"Instructions: {prompt}\n\nUser: {request.form['prompt']}\nSystem:"}
            ],
            temperature=0.5,
            presence_penalty=1,
            frequency_penalty=1
        )
    except openai.error.InvalidRequestError:
        return "Invalid request. Please try again."
    except openai.error.AuthenticationError:
        return "Invalid API key. Please try again."

    return response["choices"][0]["message"]["content"].strip()


@app.route('/verify', methods=['POST'])
def verify():
    if request.form['flag'] == get_flag():
        return "Correct!"
    else:
        return "Incorrect!"


if __name__ == '__main__':
    config: Config = Config(
        api_key_file="openai_key.txt",
        model_description_file="model_description.txt",
        model="gpt-3.5-turbo"
    )

    openai.api_key = config.api_key

    app.config.from_object(config)

    app.run(debug=True)

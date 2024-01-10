import io
import utils.ip_limiter
from flask import Blueprint, Flask, current_app, request, send_file
from flask_restx import Api, Resource, abort, fields, Namespace
import requests
from openai import OpenAI
import openai

"""
    This is the adapter to communicate with the OpenAI API
    Endpoints:
        - /openai/text-generattion - POST
        - /openai/image-generation - POST
        - /openai/tts - POST
"""

openai_namespace = Namespace("openai", description="OpenAI related operations")


def ask_question(question, pre_prompt):
    openai.api_key = current_app.config["OPENAI_API_KEY"]
    client = OpenAI()

    response = requests.post(
        f"{current_app.config['API_URL']}/log/external_api",
        json={"message": f"[OPENAI GPT] Pre-Prompt: {pre_prompt}, prompt: {question}"},
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": pre_prompt},
            {
                "role": "user",
                "content": question,
            },
        ],
        model="gpt-3.5-turbo",
        stream=True,
        max_tokens=1000,  # 0.001$ per 1000 tokens input, 0.002$ per 1000 tokens output
    )

    answer = ""
    for chunk in response:
        if (
            len(chunk.choices) > 0
            and chunk.choices[0].delta is not None
            and chunk.choices[0].delta.content is not None
        ):
            answer += chunk.choices[0].delta.content
    return answer


def create_image(prompt):
    openai.api_key = current_app.config["OPENAI_API_KEY"]
    client = OpenAI()

    response = requests.post(
        f"{current_app.config['API_URL']}/log/external_api",
        json={"message": f"[OPENAI DALL-E] Prompt: {prompt}"},
    )

    response = client.images.generate(
        # model="dall-e-3",
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


def create_tts(text):
    openai.api_key = current_app.config["OPENAI_API_KEY"]
    client = OpenAI()

    response = requests.post(
        f"{current_app.config['API_URL']}/log/external_api",
        json={"message": f"[OPENAI TTS] Prompt: {text}"},
    )

    response = client.audio.speech.create(model="tts-1", voice="echo", input=text)
    # response.stream_to_file(output_path)

    """
        TODO: save the audio!
    """
    return send_file(
        io.BytesIO(response.content),
        mimetype="audio/wav",
        as_attachment=True,
        download_name="tts.wav",
    )


generation_model = openai_namespace.model(
    "Generation",
    {
        "prompt": fields.String(required=True, description="Prompt to generate from"),
        "pre_prompt": fields.String(
            required=False, description="Pre prompt to generate from"
        ),
    },
)

tts_model = openai_namespace.model(
    "TTS", {"text": fields.String(required=True, description="Text to generate from")}
)


@openai_namespace.route("/text-generation")
@openai_namespace.doc(
    responses={200: "OK", 400: "Invalid Argument", 500: "Mapping Key Error"},
    description="Generate text from prompt",
    params={"prompt": "The prompt to generate from"},
)
class TextGenerate(Resource):
    @openai_namespace.expect(generation_model)
    @utils.ip_limiter.limit_ip_access
    def post(self):
        """
        Generate text from prompt
        """
        data = request.get_json()
        prompt = data["prompt"]
        pre_prompt = data["pre_prompt"]
        return {"text": ask_question(prompt, pre_prompt)}


@openai_namespace.route("/image-generation")
@openai_namespace.doc(
    responses={200: "OK", 400: "Invalid Argument", 500: "Mapping Key Error"},
    description="Generate image from prompt",
    params={"prompt": "The prompt to generate from"},
)
class ImageGenerate(Resource):
    @openai_namespace.expect(tts_model)
    @utils.ip_limiter.limit_ip_access
    def post(self):
        """
        Generate image from prompt
        """
        data = request.get_json()
        prompt = data["text"]
        return {"image": create_image(prompt)}


@openai_namespace.route("/tts")
@openai_namespace.doc(
    responses={200: "OK", 400: "Invalid Argument", 500: "Mapping Key Error"},
    description="Generate text to speech from prompt",
    params={"prompt": "The prompt to generate from"},
)
class TTS(Resource):
    @openai_namespace.expect(tts_model)
    @utils.ip_limiter.limit_ip_access
    def post(self):
        """
        Generate text to speech from prompt
        """
        data = request.get_json()
        prompt = data["text"]
        return create_tts(prompt)

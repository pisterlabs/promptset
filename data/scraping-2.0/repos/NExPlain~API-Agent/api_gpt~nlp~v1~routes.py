import openai
import requests
from api_gpt.services.openai_request import *
from api_gpt.nlp.v1.generation import *
from api_gpt.utils import *


@current_app.route("/plasma_ai_generation", methods=["POST"])
@token_required
def plasma_ai_generation():
    try:
        _json = request.json
        loginemail = _json["loginemail"]
        model = _json["model"]
        user_name = get_key_or_none(_json, "user_name")
        user_id = get_key_or_none(_json, "user_id")
        email = get_key_or_none(_json, "email")
        current_time = get_key_or_none(_json, "current_time")
        participants_string = get_key_or_none(_json, "participants_string")
        text = get_key_or_none(_json, "text")
        max_tokens = get_key_or_default(_json, "max_tokens", 128)
        json_response = fetch_generate_v1_response(
            user_name=user_name,
            user_id=user_id,
            email=email,
            current_time=current_time,
            participants_string=participants_string,
            text=text,
            max_tokens=max_tokens,
            model=model,
        )

        return {"message": "success", "status_code": 200, "data": json_response}

    except Exception as e:
        print("Error in plasma_ai_generation ", e, flush=True)


@current_app.route("/request_gpt_result", methods=["POST"])
@token_required
def request_gpt_result():
    try:
        _json = request.json
        loginemail = _json["loginemail"]
        prompt = _json["prompt"]
        model = _json["model"]
        max_tokens = get_key_or_default(_json, "max_tokens", 128)
        if model is None:
            model = "text-davinci-003"
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        openai_response = requests.post(
            url="https://api.openai.com/v1/completions", headers=headers, json=data
        )
        json_response = openai_response.json()
        return {"message": "success", "status_code": 200, "data": json_response}

    except Exception as e:
        print("Error in request_gpt_result ", e, flush=True)


@current_app.route("/request_chat_gpt", methods=["POST"])
@token_required
def request_chat_gpt():
    try:
        _json = request.json
        loginemail = _json["loginemail"]
        content = _json["content"]
        system = _json["system"]
        max_tokens = get_key_or_default(_json, "max_tokens", 128)
        json_response = get_chat_gpt_response(max_tokens, system, content)
        return {"message": "success", "status_code": 200, "data": json_response}

    except Exception as e:
        print("Error in request_gpt_result ", e, flush=True)

import os
import uuid
from typing import Dict

import dotenv
from flask import Blueprint, request
from flask_cors import cross_origin
from openai import OpenAI

dotenv.load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # API key 문의
from utils import execution_utils


def construct_blueprint(server_information: Dict) -> Blueprint:
    refactorization = Blueprint("refactor", __name__)

    @refactorization.post("")
    @cross_origin()
    def refactor_code():
        request_body = request.get_json()
        code = request_body.get("code", "")

        response = client.completions.create(
            model="text-davinci-003", prompt=generate_prompt(code), max_tokens=1024
        )
        full_refactored_code = response.choices[0].text.strip()

        # "public" 단어부터 시작하는 리팩토링된 코드 추출
        start_index = full_refactored_code.find("import")
        if start_index != -1:
            refactored_code = full_refactored_code[start_index:]
        else:
            refactored_code = "Refactored code not found."

        return {
            "refactored_code": refactored_code,
        }

    return refactorization


def generate_prompt(code: str):
    return f"Here is a Java code snippet:\n{code}\n\n \
        Please optimize this Java code for runtime or memory efficiency. \
        For example, using different algorithm. \
        Show me only the complete executable code include the part where libraries are imported"

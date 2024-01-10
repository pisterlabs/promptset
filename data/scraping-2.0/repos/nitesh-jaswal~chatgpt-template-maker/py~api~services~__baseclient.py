import openai
import json
from typing import Dict
from api.models.openai_models import OpenAIChatRequest, OpenAIChatResponse
from api.models.openai_models.__response import Usage

class BaseClient:

    def __init__(self, api_key: str, organization: str):
        openai.organization = organization
        openai.api_key = api_key

    def send(self, request: OpenAIChatRequest) -> OpenAIChatResponse:
        request_dict: Dict[str, str] = json.loads(request.json())
        print(request_dict)
        response = openai.ChatCompletion.create(**request_dict)
        return OpenAIChatResponse.parse_obj(response)

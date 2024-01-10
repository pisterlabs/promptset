import abc
import base64
import datetime
import hashlib
import hmac
import json
from typing import List

import openai
import requests
import tiktoken
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from openai.openai_object import OpenAIObject
from requests import Response
from rest_framework.request import Request

from apps.chat.constants import (
    AI_API_REQUEST_TIMEOUT,
    HUNYUAN_DATA_PATTERN,
    OpenAIModel,
    OpenAIUnitPrice,
)
from apps.chat.exceptions import UnexpectedError
from apps.chat.models import ChatLog, HunYuanChuck, Message

USER_MODEL = get_user_model()


# pylint: disable=R0902
class BaseClient:
    """
    Base Client for Chat
    """

    # pylint: disable=R0913
    def __init__(self, request: Request, model: str, messages: List[Message], temperature: float, top_p: float):
        self.log: ChatLog = None
        self.request: Request = request
        self.user: USER_MODEL = request.user
        self.model: str = model
        self.messages: List[Message] = messages
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.created_at: int = int()
        self.finished_at: int = int()

    @abc.abstractmethod
    def chat(self, *args, **kwargs) -> any:
        """
        Chat
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def record(self, *args, **kwargs) -> None:
        """
        Record Log
        """

        raise NotImplementedError()


class OpenAIClient(BaseClient):
    """
    OpenAI Client
    """

    @transaction.atomic()
    def chat(self, *args, **kwargs) -> any:
        self.created_at = int(timezone.now().timestamp() * 1000)
        response = openai.ChatCompletion.create(
            api_base=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        )
        for chunk in response:
            self.record(response=chunk)
            yield chunk.choices[0].delta.get("content", "")
        self.finished_at = int(timezone.now().timestamp() * 1000)
        self.post_chat()

    # pylint: disable=W0221,R1710
    def record(self, response: OpenAIObject, **kwargs) -> None:
        # check log exist
        if self.log:
            self.log.content += response.choices[0].delta.get("content", "")
            return
        # create log
        self.log = ChatLog.objects.create(
            chat_id=response.id,
            user=self.user,
            model=self.model,
            messages=self.messages,
            content="",
            created_at=self.created_at,
        )
        return self.record(response=response)

    def post_chat(self) -> None:
        if not self.log:
            return
            # calculate tokens
        encoding = tiktoken.encoding_for_model(self.model)
        self.log.prompt_tokens = len(encoding.encode("".join([message["content"] for message in self.log.messages])))
        self.log.completion_tokens = len(encoding.encode(self.log.content))
        # calculate price
        price = OpenAIUnitPrice.get_price(self.model)
        self.log.prompt_token_unit_price = price.prompt_token_unit_price
        self.log.completion_token_unit_price = price.completion_token_unit_price
        # save
        self.log.finished_at = self.finished_at
        self.log.save()
        self.log.remove_content()

    @classmethod
    def list_models(cls) -> List[dict]:
        all_models = openai.Model.list(
            api_base=settings.OPENAI_API_BASE, api_key=settings.OPENAI_API_KEY
        ).to_dict_recursive()["data"]
        supported_models = [
            {"id": model["id"], "name": str(OpenAIModel.get_name(model["id"]))}
            for model in all_models
            if model["id"] in OpenAIModel.values
        ]
        supported_models.append({"id": OpenAIModel.HUNYUAN.value, "name": str(OpenAIModel.HUNYUAN.label)})
        supported_models.sort(key=lambda model: model["id"])
        return supported_models


class HunYuanClient(BaseClient):
    """
    Hun Yuan
    """

    @transaction.atomic()
    def chat(self, *args, **kwargs) -> any:
        # log
        self.created_at = int(timezone.now().timestamp() * 1000)
        # call hunyuan api
        response = self.call_api()
        # explain completion
        completion_text = bytes()
        for chunk in response:
            completion_text += chunk
            # match hunyuan data content
            match = HUNYUAN_DATA_PATTERN.search(completion_text)
            if match is None:
                continue
            # load content
            resp_text = completion_text[match.regs[0][0] : match.regs[0][1]]
            completion_text = completion_text[match.regs[0][1] :]
            resp_text = json.loads(resp_text.decode()[6:])
            chunk = HunYuanChuck.create(resp_text)
            if chunk.error.code:
                raise UnexpectedError(detail=chunk.error.message)
            self.record(response=chunk)
            yield chunk.choices[0].delta.content
        if not self.log:
            return
        self.log.finished_at = int(timezone.now().timestamp() * 1000)
        self.log.save()
        self.log.remove_content()

    # pylint: disable=W0221,R1710
    def record(self, response: HunYuanChuck) -> None:
        # check log exist
        if self.log:
            self.log.content += response.choices[0].delta.content
            self.log.prompt_tokens = response.usage.prompt_tokens
            self.log.completion_tokens = response.usage.completion_tokens
            price = OpenAIUnitPrice.get_price(self.model)
            self.log.prompt_token_unit_price = price.prompt_token_unit_price
            self.log.completion_token_unit_price = price.completion_token_unit_price
            return
        # create log
        self.log = ChatLog.objects.create(
            chat_id=response.id,
            user=self.user,
            model=self.model,
            messages=self.messages,
            content="",
            created_at=self.created_at,
        )
        return self.record(response=response)

    def call_api(self) -> Response:
        data = {
            "app_id": settings.QCLOUD_APP_ID,
            "secret_id": settings.QCLOUD_SECRET_ID,
            "timestamp": int(timezone.now().timestamp()),
            "expired": int((timezone.now() + datetime.timedelta(minutes=5)).timestamp()),
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": 1,
        }
        message_string = ",".join(
            [f"{{\"role\":\"{message['role']}\",\"content\":\"{message['content']}\"}}" for message in self.messages]
        )
        message_string = f"[{message_string}]"
        params = {**data, "messages": message_string}
        params = dict(sorted(params.items(), key=lambda x: x[0]))
        url = (
            settings.QCLOUD_HUNYUAN_API_URL.split("://", 1)[1]
            + "?"
            + "&".join([f"{key}={val}" for key, val in params.items()])
        )
        signature = hmac.new(settings.QCLOUD_SECRET_KEY.encode(), url.encode(), hashlib.sha1).digest()
        encoded_signature = base64.b64encode(signature).decode()
        headers = {"Authorization": encoded_signature}
        resp = requests.post(
            settings.QCLOUD_HUNYUAN_API_URL, json=data, headers=headers, stream=True, timeout=AI_API_REQUEST_TIMEOUT
        )
        return resp

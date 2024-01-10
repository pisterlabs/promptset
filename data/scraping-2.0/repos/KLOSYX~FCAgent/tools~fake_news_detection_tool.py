from __future__ import annotations

import json
from pathlib import Path
from typing import Type
from urllib.parse import urljoin

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from pyrootutils import setup_root

from config import config

root = setup_root(".")


def get_core_result(text: str, image: str) -> str:
    # 构造请求参数
    params = {"image": image, "text": text}
    # 发送POST请求
    response = requests.post(
        urljoin(config.core_server_addr, "/core"),
        data=params,
    )
    # 获取响应结果
    result = response.json()
    return f"fake probability: {result['fake_prob']:.0%}\treal probability: {result['real_prob']:.0%}\n"


def load_image_content(image_path: str) -> dict:
    with open(Path(image_path)) as f:
        tweet_content = json.loads(f.read())
    return tweet_content


class FNDScheme(BaseModel):
    text: str = Field(description="Should be text content of the tweet.")
    image_path: str = Field(
        description="Should be image path of the tweet.",
        default=str(root / ".temp" / "tweet_content.json"),
    )


class FakeNewsDetectionTool(BaseTool):
    name = "fnd_tool"
    description = (
        "use this tool to get machine learning model prediction whether a tweet is true/false. "
        "CANNOT be used as the only indicator. "
        "the parameter should be `text` and `image_path`."
    )
    args_schema: type[FNDScheme] = FNDScheme

    def _run(
        self, text: str, image_path: str = str(root / ".temp" / "tweet_content.json")
    ) -> str:
        """use tweet summary as input. could be in English and Chinese."""
        tweet_content = load_image_content(image_path)
        return get_core_result(text=text, image=tweet_content["tweet_image"])

    def _arun(self, text: str) -> list[str]:
        raise NotImplementedError("This tool does not support async")

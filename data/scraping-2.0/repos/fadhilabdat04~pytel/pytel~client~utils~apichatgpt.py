# pytel < https://t.me/kastaid >
# Copyright (C) 2023-present kastaid
#
# This file is a part of < https://github.com/kastaid/pytel/ >
# Please read the GNU Affero General Public License in
# < https://github.com/kastaid/pytel/blob/main/LICENSE/ >

from typing import Optional, Any
import openai
from requests import post
from ...config import AI_KEY, AI_BASE


class PytelAI:
    """
    ChatGPT :: Open AI
    """

    def __init__(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
    ):
        self.api_key = api_key
        self.api_base = api_base

    def text(
        self,
        query: Optional[str],
    ) -> Optional[str]:
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"{query}",
                    },
                ],
            )
            if response["choices"][0][
                "message"
            ]["content"]:
                return response[
                    "choices"
                ][0]["message"][
                    "content"
                ]

        except (
            openai.error.APIError
        ) as e:
            if (
                hasattr(e, "response")
                and "detail"
                in e.response
            ):
                return e.response[
                    "detail"
                ]
            else:
                return str(e)

    def images(
        self, query: Optional[str]
    ):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        try:
            response = (
                openai.Image.create(
                    prompt=str(query),
                    n=4,
                    size="1024x1024",
                )
            )
            for image in response[
                "data"
            ]:
                return image["url"]
        except (
            openai.error.APIError
        ) as e:
            if (
                hasattr(e, "response")
                and "detail"
                in e.response
            ):
                return e.response[
                    "detail"
                ]
            else:
                return str(e)

    def tts(self, query: Optional[str]):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        strings = {
            "text": query,
            "language": "id",
        }
        response = post(
            "https://chimeragpt.adventblocks.cc/api/v1/audio/tts/generation",
            headers=headers,
            json=strings,
        ).json()
        if response["url"]:
            return response["url"]
        elif (response["detail"]) or (
            "Invalid response"
            in response
        ):
            return False

    def audio_transmitting(
        self,
        audiofile: Any,
        type_transmitting: str = None,
    ):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        if (
            type_transmitting
            == "transcribe"
        ):
            with open(
                audiofile, "rb"
            ) as au_f:
                try:
                    trans = openai.Audio.transcribe(
                        file=au_f,
                        model="whisper-1",
                        response_format="text",
                        language="id",
                    )
                    if trans["text"]:
                        return trans[
                            "text"
                        ]
                except (
                    openai.error.APIError
                ) as e:
                    if (
                        hasattr(
                            e,
                            "response",
                        )
                        and "detail"
                        in e.response
                    ):
                        return (
                            e.response[
                                "detail"
                            ]
                        )
                    else:
                        return str(e)

        if (
            type_transmitting
            == "translate"
        ):
            with open(
                audiofile, "rb"
            ) as au_f:
                try:
                    trans = openai.Audio.translate(
                        file=au_f,
                        model="whisper-1",
                        response_format="text",
                        language="id",
                    )
                    if trans["text"]:
                        return trans[
                            "text"
                        ]
                except (
                    openai.error.APIError
                ) as e:
                    if (
                        hasattr(
                            e,
                            "response",
                        )
                        and "detail"
                        in e.response
                    ):
                        return (
                            e.response[
                                "detail"
                            ]
                        )
                    else:
                        return str(e)


ChatGPT = PytelAI(
    api_key=AI_KEY,
    api_base=AI_BASE,
)

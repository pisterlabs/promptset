import asyncio
from io import BytesIO
import json
import random

from typing import List, Literal, Dict, Union
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletion,
)
from openai._exceptions import APIStatusError
from httpx import AsyncClient
from pydantic import BaseModel
from loguru import logger

from .types import (
    Channel,
    Session,
    ToolCall,
    ToolCallConfig,
    ToolCallResponse,
    ToolCallRequest,
    FuncContext,
)
from .function import ToolsFunction


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        channels: List[Channel],
        tool_func: ToolsFunction,
        default_model: str = "gpt-3.5-turbo",
    ):
        self.channels = channels
        self.http_client = AsyncClient(base_url=base_url, follow_redirects=True)
        self.tool_func = tool_func
        self.default_model = default_model

    def init_client(self, channel: Channel):
        client = AsyncOpenAI(**channel.dict(), http_client=self.http_client)
        return client

    @property
    def client(self):
        channel = random.choice(self.channels)
        return self.init_client(channel)

    async def chat(
        self,
        session: Session,
        prompt: str = "",
        model: str = "",
        image_url: str = "",
        tool_choice: Literal["none", "auto"] = "auto",
    ) -> List[Union[ToolCallRequest, ChatCompletionMessage]]:
        if not model:
            model = self.default_model
        if prompt:
            session.messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(prompt + f"\n![img]({image_url})") if image_url else prompt,
                )
            )
        results = await self.chat_completions(
            session=session, model=model, tool_choice=tool_choice
        )
        return results

    async def chat_completions(
        self,
        session: Session,
        model="gpt-3.5-turbo",
        tool_choice: Literal["none", "auto"] = "auto",
    ) -> List[Union[ToolCallRequest, ChatCompletionMessage]]:
        """
        该函数用于生成聊天的完成内容。

        参数:
        session (Session): 当前的会话对象。
        model (str, 可选): 使用的模型名称，默认为"gpt-3.5-turbo"。
        tool_choice (Literal["none", "auto"], 可选): 工具选择，默认为"auto"。

        返回:
        results (list): 包含完成内容的列表。

        """
        # 检查模型名称中是否包含"vision"
        vision = model.count("vision") > 0
        messages = session.get_messages()
        if vision:
            messages = [messages[-1]]
            session.messages.pop()
        max_retry = 3
        for i in range(max_retry):
            try:
                # 创建聊天完成内容
                chat_completion = await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    tool_choice=None if vision or tool_choice == "none" else tool_choice,
                    tools=None
                    if vision or tool_choice == "none"  # 省 Tokens
                    else self.tool_func.tools_info(),
                    user=session.user,
                    max_tokens=1024 if vision else None,
                )
                break
            except APIStatusError as e:
                logger.error(f"请求聊天出错: {e}")
                if i == max_retry - 1:
                    return [
                        ChatCompletionMessage(
                            role="assistant",
                            content=f"请求聊天出错: {e.message}",
                        )
                    ]
            except Exception as e:
                logger.error(f"请求聊天出错: {e}")
        else:
            return [
                ChatCompletionMessage(
                    role="assistant",
                    content="请求聊天出错",
                )
            ]
        return self.make_chat_completion_results(session, chat_completion)

    def make_chat_completion_results(
        self, session: Session, chat_completion: ChatCompletion
    ):
        logger.info(f"chat_comletion: {chat_completion}")
        results = []
        choices = chat_completion.choices
        for choice in choices:
            if choice.message.role == "":
                choice.message.role = "assistant"
            # 将选择的消息添加到结果列表中
            results.append(choice.message)
            if choice.message.tool_calls:
                # 清空消息内容，防止OpenAI奇怪的报错
                choice.message.content = ""

                # 遍历每个工具调用
                for tool_call in choice.message.tool_calls:
                    # 将工具调用请求添加到结果列表中
                    results.append(self.make_tool_request(session, tool_call))

            # 如果消息中包含函数调用
            if choice.message.function_call:
                # 清空消息内容，防止OpenAI奇怪的报错
                choice.message.content = ""

                # 调用函数
                task = self.tool_func.call_function(
                    function_call=choice.message.function_call,
                    session=session,
                )

                # 将函数调用请求添加到结果列表中
                results.append(
                    ToolCallRequest(
                        tool_call=choice.message.function_call,
                        func=task,
                        config=self.tool_func.tool_config[tool_call.function.name],
                    )
                )

            # 将选择的消息添加到会话的消息列表中
            session.messages.append(choice.message)
        results.append(chat_completion.usage)
        return results

    def make_tool_request(
        self, session: Session, tool_call: ChatCompletionMessageToolCall
    ):
        config = self.tool_func.tool_config.get(tool_call.function.name)
        task = self.tool_func.call_tool(
            tool_call=tool_call,
            ctx=FuncContext[type(config)](
                session=session,
                openai_client=self.client,
                http_client=self.http_client,
                config=config,
            ),
        )
        return ToolCallRequest(
            tool_call=tool_call,
            func=task,
            config=config,
        )

    async def tts(
        self,
        input: str,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "shimmer",
        speed: float = 1.0,
        ctx: FuncContext[ToolCallConfig] = None,
    ):
        """
        Generates audio from the input text. Can produce a method of speaking to be used in a
        voice application.

        Args:
          input: The text to generate audio for. The maximum length is 4096 characters.

          model:
              One of the available [TTS models](https://platform.openai.com/docs/models/tts):
              `tts-1` or `tts-1-hd`

          voice: The voice to use when generating the audio. Supported voices are `alloy`,
              `echo`, `fable`, `onyx`, `nova`, and `shimmer`.

          speed: The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is
              the default.
        """
        logger.info(f"tts: {input} {model} {voice} {speed}")
        resp = ToolCallResponse(
            name="tts",
            content_type="audio",
            content=None,
            data="success to generate audio, it has been display.",
        )
        if isinstance(speed, str):
            speed = float(speed)
        try:
            record = await self.client.audio.speech.create(
                input=input, model=model, voice=voice, speed=speed
            )
        except APIStatusError as e:
            logger.error(f"TTS: {e}")
            resp.data = f"failed to generate audio, {e.message}"
            return resp
        except Exception as e:
            logger.error(f"TTS: {e}")
            resp.data = f"failed to generate audio, {e}"
            return resp
        resp.content = record.content
        return resp

    async def gen_image(
        self,
        prompt: str,
        model: Literal["dall-e-2", "dall-e-3"] = "dall-e-3",
        quality: Literal["standard", "hd"] = "standard",
        size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        style: Literal["vivid", "natural"] = "vivid",
        ctx: FuncContext[ToolCallConfig] = None,
    ):
        """
        Creates an image given a prompt.

        Args:
          prompt: A text description of the desired image(s). The maximum length is 1000
              characters for `dall-e-2` and 4000 characters for `dall-e-3`.

          model: The model to use for image generation.

          quality: The quality of the image that will be generated. `hd` creates images with finer
              details and greater consistency across the image. This param is only supported
              for `dall-e-3`.

          size: The size of the generated images. Must be one of `256x256`, `512x512`, or
              `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or
              `1024x1792` for `dall-e-3` models.

          style: The style of the generated images. Must be one of `vivid` or `natural`. Vivid
              causes the model to lean towards generating hyper-real and dramatic images.
              Natural causes the model to produce more natural, less hyper-real looking
              images. This param is only supported for `dall-e-3`.
        """
        logger.info(f"gen_image: {prompt} {model} {quality} {size} {style}")
        resp = ToolCallResponse(
            name="gen_image",
            content_type="openai_image",
            content=None,
            data="failed to generate image",
        )
        try:
            image_resp = await self.client.images.generate(
                prompt=prompt,
                n=1,
                response_format="url",
                model=model,
                quality=quality,
                size=size,
                style=style,
            )
        except APIStatusError as e:
            logger.error(f"DALL-E: {e}")
            resp.data = f"failed to generate image, {e.message}"
            return resp
        except Exception as e:
            logger.error(f"DALL-E: {e}")
            resp.data = f"failed to generate image, {e}"
            return resp
        if image_resp.created:
            data = image_resp.data[0]
            resp.data = f"success generate image and it had been display, here is revised prompt of this image: {data.revised_prompt}" 
            resp.content = data
        return resp

    async def vision(
        self,
        url: str,
        text: str = "Analyze this picture",
        ctx: FuncContext[ToolCallConfig] = None,
    ):
        """
        This function is designed to analyze an image using a chat completion and return the analysis results.

        Args:
            text (str): The text to be used as context for the image analysis.
            url (str): The URL of the image to be analyzed.

        Returns:
            ToolCallResponse: The response from the tool call, containing the analysis result.

        Raises:
            APIStatusError: If there is an error with the API status.
            Exception: If there is a general error.
        """
        logger.info(f"Vision: {text} {url}")
        resp = ToolCallResponse(
            name="vision",
            content_type="str",
            content=None,
            data="failed to analyze image",
        )
        try:
            analyze_resp = await self.client.chat.completions.create(
                messages=[
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(
                                text=text,
                                type="text",
                            ),
                            ChatCompletionContentPartImageParam(
                                image_url={
                                    "url": url
                                },
                                type="image_url",
                            ),
                        ],
                    ),
                ],
                model="gpt-4-vision-preview",
                max_tokens=1024,
            )
        except APIStatusError as e:
            logger.error(f"Vision: {e}")
            resp.data = f"failed to analyze image, {e.message}"
            return resp
        except Exception as e:
            logger.error(f"Vision: {e}")
            resp.data = f"failed to analyze image, {e}"
            return resp
        if analyze_resp.created:
            data = analyze_resp.choices[0].message
            resp.data = data.content
        return resp

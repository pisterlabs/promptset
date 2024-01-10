import json
from typing import TYPE_CHECKING, Union
from openai import AsyncOpenAI, OpenAI, AsyncAzureOpenAI

from ..models import *
from ..schemas import *
from config import settings
from modules.messages.schemas import AssistantMessage

if TYPE_CHECKING:
    from core.services.message_handler import MessageHandler
    from modules.messages.services.message_chain import MessageChain


class LLMConnector:
    def __init__(
        self,
        context: "MessageChain",
        message_handler: "MessageHandler",
        api_type: str = "openai",
    ):
        self.api_type = api_type
        self.message_handler = message_handler
        self.context = context

        self.create_client()

    def create_client(self):
        # Azure config
        if self.api_type == "azure":
            api_key = settings.chat_models.get("azure_openai_key", None)
            base_url = settings.chat_models.get("azure_openai_endpoint", None)
            api_version = settings.chat_models.get(
                "azure_openai_api_version", "2023-05-15"
            )

            if not api_key or not base_url:
                raise ValueError(
                    "AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set"
                )

            self.client = AsyncAzureOpenAI(
                api_key=api_key, base_url=base_url, api_version=api_version
            )
        # Custom config
        elif self.api_type == "custom":
            base_url = settings.chat_models.get("custom_endpoint", None)

            if not base_url:
                raise ValueError(
                    "CUSTOM_ENDPOINT must be set in environment variables."
                )

            self.client = AsyncOpenAI(base_url=base_url)

        else:
            api_key = settings.chat_models.get("openai_api_key", None)
            base_url = "https://api.openai.com/v1"

            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def create_chat(
        self,
        messages,
        conversation_id,
        model="gpt-4",
        temperature=0.7,
        top_p=1,
        n=1,
        stream=True,
        functions=None,
        max_tokens=0,
        stop=None,
        presence_penalty=0,
        frequency_penalty=0,
    ) -> Union[AssistantMessage, None]:
        model_key = "deployment_id" if self.api_type == "azure" else "model"
        for attempt in range(3):
            try:
                params = {
                    model_key: model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                    "stream": stream,
                    "stop": stop,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                }

                if functions:
                    params["functions"] = functions

                if max_tokens and max_tokens > 0:
                    params["max_tokens"] = max_tokens

                response = await self.client.chat.completions.create(**params)

                if not stream:
                    serialized_response = response.model_dump()
                    return serialized_response["choices"][0]["message"]["content"]
                else:
                    collected_tokens = ""
                    function_name = ""
                    function_arguments = ""
                    async for response_chunk in response:
                        try:
                            chunk = response_chunk.model_dump()
                            delta = chunk["choices"][0]["delta"]

                            if delta:
                                collected_tokens += delta.get("content", "") or ""
                                function_call = delta.get("function_call")

                                if function_call:
                                    function_name = (
                                        function_call.get("name") or function_name
                                    )
                                    function_arguments += (
                                        function_call.get("arguments", "") or ""
                                    )

                            ai_message = AssistantMessage(
                                conversation_id=conversation_id,
                                content=collected_tokens,
                                status="incomplete",
                            )

                            await self.message_handler.send_message_to_ui(
                                message=ai_message.content,
                                conversation_id=ai_message.conversation_id,
                                status=ai_message.status,
                                save_to_db=False,
                            )
                        except Exception as e:
                            print("Error in chunk: ", chunk)
                            print(e)

                    function_call = (
                        {
                            "name": function_name,
                            "arguments": json.loads(function_arguments),
                        }
                        if function_name
                        else None
                    )

                    ai_message.status = "complete"
                    ai_message.metadata = self.context.get_metadata()
                    ai_message.function_call = function_call
                    ai_message.xray = {"messages": self.context.get_chain_as_dict()}

                    await self.message_handler.send_message_to_ui(
                        message=ai_message.content,
                        conversation_id=ai_message.conversation_id,
                        metadata=ai_message.metadata,
                        function_call=ai_message.function_call,
                        status=ai_message.status,
                        xray=ai_message.xray,
                        save_to_db=False,
                    )

                    return ai_message
            except Exception as e:
                await self.message_handler.send_alert_to_ui(
                    message=str(e).replace("OpenAI", "chat model"), type="error"
                )
                print("An exception occured: ", e)
                return None

        await self.message_handler.send_alert_to_ui(
            message="Error connecting to chat model!", type="error"
        )
        print("Failed after 3 retries.")
        return None


def get_embeddings(doc: str):
    api_key = settings.chat_models.get("openai_api_key", None)
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=doc, model="text-embedding-ada-002")
    serialized = response.model_dump()
    embeddings = serialized["data"][0]["embedding"]
    return embeddings

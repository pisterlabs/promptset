import abc
import os
import logging

from notionai import NotionAI
from typing import List, Optional, Dict
from notionai.enums import ToneEnum, TranslateLanguageEnum, PromptTypeEnum
import openai

LOGGER = logging.getLogger("ask_ai")

# PYLLM_PROVODERS = [
#     "openai",
#     "anthropic",
#     "ai21",
#     "cohere",
#     "alephalpha",
#     "huggingface_hub",
#     "google",
# ]
# OTHER_PROVIDERS = ["notionai", "bingchat", "openai"]
# PROVODERS = PYLLM_PROVODERS + OTHER_PROVIDERS
HUGCHAT_LLMS = ['OpenAssistant/oasst-sft-6-llama-30b-xor', 'meta-llama/Llama-2-70b-chat-hf']


class AIProvider(abc.ABC):

    @abc.abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        pass

    def complete_and_remove_prompt(self, prompt: str, **kwargs) -> str:
        result = self.complete(prompt, **kwargs)
        if result.startswith(prompt):
            result = result[len(prompt):]
        return result

    def change_tone(self, tone: str, context: str):
        promt = f"Change the tone to {tone}:\n{context}"
        return self.complete_and_remove_prompt(promt)

    def improve_writing(self, context):
        promt = f"Improve the writing:\n{context}"
        return self.complete_and_remove_prompt(promt)

    def continue_writing(self, context, page_title=""):
        promt = f"Continue writing:\n{context}"
        return self.complete_and_remove_prompt(promt)

    def translate(self, language, context):
        promt = f"Translate to {language}:\n{context}"
        return self.complete_and_remove_prompt(promt)

    def summarize(self, context):
        promp = f"Summarize:\n{context}"
        return self.complete_and_remove_prompt(promp)

    @classmethod
    def _build_one(cls, model_provoder: str):
        parts = model_provoder.split("_", 1)
        provider = parts[0]
        if provider == "openai":
            return OpenAIProvider(
                parts[1]) if len(parts) == 2 else OpenAIProvider()
        # elif provider in PYLLM_PROVODERS:
        #     model = parts[1]
        #     return PyLLMProvider(provider, model)
        elif provider == "hugchat":
            model = model_provoder.split("_", 1)[1]
            return HugChatProvider(model)
        elif provider == "notionai":
            return NotionAIProvider()
        elif provider == "bingchat":
            style = model_provoder.split("_", 1)[1]
            return BingChatProvider(style)
        else:
            raise Exception(f"not support provider {provider}")

    @classmethod
    def build(cls, provoders: List[str]):
        LOGGER.debug(f"Providers: {provoders}")
        if len(provoders) == 1:
            return AIProvider._build_one(provoders[0])
        else:
            return MultiProvider(provoders)

class HugChatProvider(AIProvider):
    def __init__(self, model):
        self.name = f"hugchat_{model}"
        self.model = model
        HUGCHAT_EMAIL = os.getenv("HUGCHAT_EMAIL")
        HUGCHAT_PASSWORD = os.getenv("HUGCHAT_PASSWORD")
        HUGCHAT_COOKIE_DIR = os.getenv("HUGCHAT_COOKIE_DIR", ".hugchat_cookies")
        if not HUGCHAT_EMAIL:
            LOGGER.error("HUGCHAT_EMAIL is not set")
        if not HUGCHAT_PASSWORD:
            LOGGER.error("HUGCHAT_PASSWORD is not set")
        logging.debug(
            f"Create HugChatProvider with email {HUGCHAT_EMAIL}"
        )
        from hugchat import hugchat
        from hugchat.login import Login
        # login
        sign = Login(HUGCHAT_EMAIL, HUGCHAT_PASSWORD)
        cookie_file_path = os.path.join(HUGCHAT_COOKIE_DIR, f"{HUGCHAT_EMAIL}.json")
        if not os.path.exists(cookie_file_path):
            cookies = sign.login()
            sign.saveCookiesToDir(HUGCHAT_COOKIE_DIR)
        else:
            # load cookies from usercookies/<email>.json
            sign = Login(HUGCHAT_EMAIL, None)
            cookies = sign.loadCookiesFromDir(HUGCHAT_COOKIE_DIR) 

        # Create a ChatBot
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        self.chatbot.active_model = model

    def complete(self,
                 prompt: str,
                # web_search: bool=False,
                temperature: float=0.9,
                top_p: float=0.95,
                repetition_penalty: float=1.2,
                top_k: int=50,
                truncate: int=1024,
                watermark: bool=False,
                max_new_tokens: int=1024,
                stop: list=["</s>"],
                return_full_text: bool=False,
                stream: bool=True,
                use_cache: bool=False,
                is_retry: bool=False,
                retry_count: int=5,
                 **kwargs) -> str:
        return self.chatbot.chat(
            prompt,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            truncate=truncate,
            watermark=watermark,
            max_new_tokens=max_new_tokens,
            stop=stop,
            return_full_text=return_full_text,
            stream=stream,
            use_cache=use_cache,
            is_retry=is_retry,
            retry_count=retry_count,
        )


class OpenAIProvider(AIProvider):

    def __init__(self, model='gpt-3.5-turbo', api_key=None):
        self.model = model
        self.name = f"openai_{model}"
        self.api_key = api_key if api_key is not None else os.getenv(
            'OPENAI_API_KEY')

        # Set the OpenAI API key
        openai.api_key = self.api_key
        self.client = openai.ChatCompletion if self.is_chat_model else openai.Completion

    @property
    def is_chat_model(self) -> bool:
        return self.model.startswith("gpt")

    def _prepapre_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: str = None,
        temperature: float = 0,
        max_tokens: int = 30000,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]

            if history:
                messages = [*history, *messages]

            if system_message:
                messages = [{
                    "role": "system",
                    "content": system_message
                }, *messages]

            model_inputs = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs,
            }
        else:
            if history:
                raise ValueError(
                    f"history argument is not supported for {self.model} model"
                )

            if system_message:
                raise ValueError(
                    f"system_message argument is not supported for {self.model} model"
                )

            model_inputs = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs,
            }
        return model_inputs

    def complete(self,
                 prompt: str,
                 history: Optional[List[dict]] = None,
                 system_message: Optional[List[dict]] = None,
                 temperature: float = 0,
                 max_tokens: int = 300,
                 **kwargs) -> str:
        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=None,
            system_message=None,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        response = self.client.create(model=self.model,
                                      **model_inputs,
                                      **kwargs)
        if self.is_chat_model:
            return response.choices[0].message.content.strip()
        return response.choices[0].text.strip()


class NotionAIProvider(AIProvider):

    def __init__(self):
        self.name = "notionai"
        TOKEN = os.getenv("NOTION_TOKEN")
        SPACE_ID = os.getenv("NOTION_SPACE_ID")
        if not TOKEN:
            LOGGER.error("NOTION_TOKEN is not set")
        if not SPACE_ID:
            LOGGER.error("NOTION_SPACE_ID is not set")
        logging.debug(
            f"Create NotionAIProvider with token {TOKEN} and space id {SPACE_ID}"
        )
        self.ai = NotionAI(TOKEN, SPACE_ID)

    def complete(self, prompt: str, **kwargs) -> str:
        return self.ai.writing_with_prompt(PromptTypeEnum.continue_writing,
                                           context=prompt,
                                           **kwargs)

    def change_tone(self, tone: str, context: str):
        tone_enum = ToneEnum(tone)
        return self.ai.translate(tone_enum, context)

    def improve_writing(self, context):
        return self.ai.improve_writing(context)

    def continue_writing(self, context, page_title=""):
        return self.ai.continue_write(context)

    def translate(self, language, context):
        language_enum = TranslateLanguageEnum(language)
        return self.ai.translate(language_enum, context)

    def summarize(self, context):
        return self.ai.summarize(context)


class BingChatProvider(AIProvider):

    def __init__(self, style):
        from sydney import SydneyClient

        self.name = "bingchat"
        LOGGER.debug(f"Create BingChatProvider with style {style}")
        self.sydney = SydneyClient(style=style)

    async def _async_complete(self, prompt: str, **kwargs) -> str:
        await self.sydney.start_conversation()
        result = await self.sydney.ask(prompt, citations=False)
        # self.sydney.reset_conversation()
        return result

    def complete(self, prompt: str, **kwargs) -> str:
        import asyncio
        answer = asyncio.run(self._async_complete(prompt, **kwargs))
        return answer


# class PyLLMProvider(AIProvider):

#     def __init__(self, provider: str, model: str):
#         import llms
#         self.name = f"{provider}_{model}"
#         LOGGER.debug(
#             f"Create PyLLMProvider with provider {provider} and model {model}")
#         self.ai = llms.init(model)

#     def complete(self, prompt: str, **kwargs) -> str:
#         return self.ai.complete(prompt, **kwargs).text


class MultiProvider(AIProvider):

    def __init__(self, providers: List[str]):
        self.providers = [AIProvider._build_one(p) for p in providers]

    def complete(self, prompt: str, **kwargs) -> str:
        results = []
        for provider in self.providers:
            results.append(f"{provider.name}:")
            results.append(provider.complete(prompt, **kwargs))
        return "\n".join(results)
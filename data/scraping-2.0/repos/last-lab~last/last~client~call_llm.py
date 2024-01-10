import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from last.llms.alles_llm import AllesChatLLM
from typing import List, Dict, Union, Optional, TypeVar, Tuple


CHAT_LLM_GPT = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-1106-preview",
]

ALLES_CHAT_LLM = [
    "alles-minimax",
    "alles-chatgpt",
    "alles-gpt4",
    "alles-palm",
    "alles-claude",
    "alles-wenxin",
    "alles-spark",
    "puyu",
    "jieyue",
    "tigerbot",
    "mita",
    "wuya",
    "soul",
    "eastmoney",
    "huazang",
    "kkbot",
    "wangyi",
    "bilibili",
    "infchat",
    "caozhi",
    "xiaohongshu",
    "shangtang",
    "claude2",
    "minimax",
    "erniebot",
    "warrenq",
    "giantgpt",
    "ruyichat",
    "midu",
    "yuewriter",
    "squirrel",
    "xiaohui",
    "bigsea",
    "starbitech",
    "wind",
]


async def call_llm(model, temperature, system_prompt, human_prompt, **kwargs):
    if model in CHAT_LLM_GPT:
        chat = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=kwargs["maximum_length"],
            model_kwargs={
                "stop": kwargs["stop_sequence"],
                "top_p": kwargs["top_p"],
                "frequency_penalty": kwargs["frequence_penalty"],
                "presence_penalty": kwargs["presence_penalty"],
            },
            max_retries=2**3,   ## 重试次数 ( langchain 默认指数退避: Wait 2^x * 1 second between each retry starting with 4 seconds, then up to 10 seconds, then 10 seconds afterwards)
        )
        messages = []
        if not (system_prompt is None or system_prompt == ""):
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=human_prompt))

        resultMessage_BaseMessage = await chat.apredict_messages(
            messages,
        )
        output = resultMessage_BaseMessage.content
    elif model.lower() in ALLES_CHAT_LLM:
        chat = AllesChatLLM(
            model=model,
            temperature=temperature,
            max_tokens=kwargs["maximum_length"],
            model_kwargs={
                "stop": kwargs["stop_sequence"],
                "top_p": kwargs["top_p"],
                "frequency_penalty": kwargs["frequence_penalty"],
                "presence_penalty": kwargs["presence_penalty"],
            },
        )

        messages = []
        if not (system_prompt is None or system_prompt == ""):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": human_prompt})

        output = await chat(
            prompt="",
            messages=messages,
        )
    else:
        raise NotImplementedError()

    return "".join(list(output))


async def generate(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    maximum_length: Optional[int] = 10000,
    temperature: Optional[float] = 0.9,
    stop_sequence: Optional[str] = None,
    top_p: Optional[float] = 0.9,
    frequence_penalty: Optional[float] = 0.0,
    presence_penalty: Optional[float] = 0.0,
) -> str:
    try:
        outputs = await call_llm(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,  # TODO 目前不支持system prompt
            human_prompt=prompt.replace('"','').replace("'",''), #防止题目中有英文引号
            maximum_length=maximum_length,
            stop_sequence=stop_sequence,
            top_p=top_p,
            frequence_penalty=frequence_penalty,
            presence_penalty=presence_penalty,
        )
    except Exception as ex:
        import traceback

        traceback.print_exc()
        # todo: 返回带 error 字段的 dic; 当前依靠 phase str; 或者抛出error?
        return "error:" + str(ex)
    return str(outputs)


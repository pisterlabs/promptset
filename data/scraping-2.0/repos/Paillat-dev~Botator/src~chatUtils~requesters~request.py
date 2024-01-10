import discord
from src.chatUtils.requesters.openaiChat import openaiChat
from src.chatUtils.requesters.openaiText import openaiText
from src.chatUtils.requesters.llama import llama
from src.chatUtils.requesters.llama2 import llama2
from src.chatUtils.requesters.claude import claude


class ModelNotFound(Exception):
    pass


async def request(
    model: str,
    prompt: list[dict] | str,
    openai_api_key: str,
    funtcions: list[dict] = None,
    custom_temp: float = 1.2,
):
    if model == "gpt-3.5-turbo":
        return await openaiChat(
            messages=prompt,
            openai_api_key=openai_api_key,
            functions=funtcions,
            model=model,
            temperature=custom_temp,
        )
    elif model == "text-davinci-003":
        #        return await openaiText(prompt=prompt, openai_api_key=openai_api_key)
        raise NotImplementedError("This model is not supported yet")
    elif model == "text-llama":
        return await llama(prompt=prompt)
    elif model == "text-llama2":
        #        return await llama2(prompt=prompt)
        raise NotImplementedError("This model is not supported yet")
    elif model == "claude":
        return await claude(messages=prompt)
    else:
        raise ModelNotFound(f"Model {model} not found")

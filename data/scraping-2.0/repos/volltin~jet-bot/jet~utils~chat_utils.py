import os
import asyncio
import logging
import datetime

import dotenv
from prompts import CHAT_SYSTEM_MESSAGE
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)


def get_current_model(model_name=None):
    if model_name:
        return model_name
    return os.getenv("OPENAI_MODEL_NAME", "unknown")


def get_all_models():
    openai_allowed_models = os.getenv("OPENAI_ALLOWED_MODELS", "")
    return list(set(openai_allowed_models.split(",") + [get_current_model()]))


def get_chat(**kwargs):
    openai_api_type = os.environ.get("OPENAI_API_TYPE", "openai")

    if "max_tokens" in kwargs:
        max_tokens = kwargs["max_tokens"]
        max_tokens = max_tokens if max_tokens > 0 else None
        kwargs["max_tokens"] = max_tokens

    model_name = None
    if "model_name" in kwargs:
        model_name = kwargs["model_name"]

    if openai_api_type == "azure":
        if "deployment_name" not in kwargs:
            kwargs["deployment_name"] = get_current_model(model_name)
        chat = AzureChatOpenAI(**kwargs)
        return chat
    else:
        if "model" not in kwargs:
            kwargs["model"] = get_current_model(model_name)
        chat = ChatOpenAI(**kwargs)
        return chat


def get_chat_system_message():
    # tz: Beijing, format %Y-%m-%d %H:%M:%S
    ts = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return CHAT_SYSTEM_MESSAGE.format(time=ts)


def make_langchain_history(gradio_history, message=None, system_message=None):
    """
    Make a history in the format of langchain from gradio history
    """
    history_langchain_format = []

    # append the system message
    if system_message:
        history_langchain_format.append(SystemMessage(content=system_message))

    # append the history
    # gradio_history: List[Tuple[Optional[str], Optional[str]]]
    # e.g. [(None, 'human message'), ('ai message', None)]
    for human, ai in gradio_history:
        if human:
            history_langchain_format.append(HumanMessage(content=human))
        if ai:
            history_langchain_format.append(AIMessage(content=ai))

    # append the current message
    if message:
        history_langchain_format.append(HumanMessage(content=message))

    return history_langchain_format


def generate_new_messages(
    message, history, system_message=None, temperature=1.0, max_tokens=0
):
    chat = get_chat(temperature=temperature, max_tokens=max_tokens)
    history_langchain_format = make_langchain_history(
        gradio_history=history, message=message, system_message=system_message
    )

    # log the actual history
    logging.info("History: %r", history_langchain_format)

    response = chat(history_langchain_format)

    new_messages = [response]
    return new_messages


async def agenerate_new_text(
    message,
    history,
    return_history=False,
    model_name=None,
    system_message=None,
    temperature=1.0,
    max_tokens=0,
):
    messages = make_langchain_history(
        gradio_history=history, message=message, system_message=system_message
    )

    if return_history:
        # append pending bot message
        history[-1][1] = ""

    # log the actual history
    logging.info("Messages: %r", messages)

    handler = AsyncIteratorCallbackHandler()

    async def wrap_done(fn, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            logging.error("Exception: %r", e)
        finally:
            event.set()  # Signal the aiter to stop.

    chat = get_chat(
        streaming=True,
        callbacks=[handler],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    task = asyncio.create_task(
        wrap_done(chat.agenerate(messages=[messages]), handler.done)
    )

    content = ""
    async for token in handler.aiter():
        content += token
        if return_history:
            # only update the bot message in the last item
            history[-1][1] += token
            yield history
        else:
            yield content

    await task

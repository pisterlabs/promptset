import logging
import os
from typing import List, Union

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.ctransformers import CTransformers
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from src import CFG

logging.basicConfig(level=logging.INFO)


def load_llama2() -> CTransformers:
    """Load Llama-2 model."""
    logging.info("Loading llama2 model ...")
    model = CTransformers(
        model=os.path.join(CFG.MODELS_DIR, CFG.LLAMA2.MODEL_PATH),
        model_type=CFG.LLAMA2.MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
            "repetition_penalty": CFG.REPETITION_PENALTY,
            "context_length": CFG.CONTEXT_LENGTH,
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    logging.info("Model loaded")
    return model


def llama2_prompt(messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> str:
    """Convert the messages to Llama2 compliant format."""
    messages = _convert_langchainschema_to_dict(messages)

    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    BOS = "<s>"
    EOS = "</s>"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe. Please ensure that your responses \
are socially unbiased and positive in nature. If a question does not make any sense, \
or is not factually coherent, explain why instead of answering something not correct."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages

    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
    return "".join(messages_list)


def _convert_langchainschema_to_dict(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
) -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    _messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            _messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            _messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            _messages.append({"role": "assistant", "content": message.content})
    return _messages

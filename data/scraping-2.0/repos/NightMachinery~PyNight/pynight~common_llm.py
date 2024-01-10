from brish import z, zp
import llm
from collections import defaultdict

from pynight.common_clipboard import clipboard_copy
from pynight.common_bells import bell_gpt
from pynight.common_openai import openai_key_get


llm_models = dict()
conversations = dict()


def chat(
    msg,
    *,
    model="gpt-3.5-turbo",
    conversation=None,
    reset_conversation=False,
    temperature=None,
    return_mode="copy",
    end="\n-------",
    bell=None,
    openai_key=None,
    **kwargs,
):
    if openai_key is None:
        openai_key = openai_key_get()

    model_name = model
    if model_name in llm_models:
        model = llm_models[model_name]
    else:
        model = llm.get_model(model_name)
        model.key = openai_key

        llm_models[model_name] = model

    conversation_name = conversation
    if conversation_name is None:
        conversation = model.conversation()
    else:
        reset_conversation = (
            reset_conversation
            or conversation_name not in conversations
            or conversations[conversation_name] is None
        )
        if reset_conversation:
            conversations[conversation_name] = model.conversation()

        conversation = conversations[conversation_name]

    if temperature is not None:
        kwargs["temperature"] = temperature

    response = conversation.prompt(
        msg,
        **kwargs,
    )

    print(
        f"{conversation.model}, Temperature: {temperature}\n\tConversation(len={len(conversation.responses)}): {conversation_name}->{conversation.id}\n"
    )
    for chunk in response:
        print(chunk, end="")
    print(end, end="")

    if bell:
        bell_gpt()

    if return_mode == "response":
        return response
    elif return_mode == "copy":
        clipboard_copy(response.text())
    else:
        return None

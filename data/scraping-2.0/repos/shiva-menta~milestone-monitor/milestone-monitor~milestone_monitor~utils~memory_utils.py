from typing import List

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import messages_from_dict, messages_to_dict, HumanMessage


def dict_to_memory(memory_dict: List[dict], k=3) -> ConversationBufferWindowMemory:
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        ai_prefix="AI",
        human_prefix="User",
        input_key="input",
        return_messages=True,
        k=k,
    )
    buffer = messages_from_dict(memory_dict)

    last_added_message_type = ""
    current_human_message = ""
    current_ai_message = ""

    print(">>> Memory buffer:")
    print(buffer)

    for message in buffer:
        # Encountered a human message
        if isinstance(message, HumanMessage):
            if last_added_message_type == "ai":
                memory.save_context(
                    {"input": current_human_message}, {"output": current_ai_message}
                )
                current_human_message = ""
                current_ai_message = ""

            last_added_message_type = "human"
            current_human_message += message.content

        # Encountered an AI message
        else:
            if not last_added_message_type:
                memory.save_context({"input": ""}, {"output": current_ai_message})

            last_added_message_type = "ai"
            current_ai_message += message.content

    if last_added_message_type == "ai":
        memory.save_context(
            {"input": current_human_message}, {"output": current_ai_message}
        )
    elif last_added_message_type == "human":
        memory.save_context({"input": current_human_message}, {"output": ""})

    return memory


def memory_to_dict(memory: ConversationBufferWindowMemory) -> List[dict]:
    return messages_to_dict(memory.buffer)


def create_main_memory(k=3) -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        ai_prefix="AI",
        human_prefix="User",
        input_key="input",
        return_messages=True,
        k=k,
    )

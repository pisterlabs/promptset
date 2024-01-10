import time
import math
import datetime

from langchain.schema import (
    LLMResult,
    messages_from_dict, messages_to_dict
)

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from openai.error import Timeout as OpenAITimeoutError

TIMEOUT = 120
TIMEOUT_RETRY = 4
PREV_CONV_IN_TOPIC = 3
RANDOM_CONV_IN_TOPIC = 3
MAX_CONV_LENGTH = 60
DATE_FORMAT = "%Y-%m-%d"

def get_today():
    return datetime.datetime.now().strftime(DATE_FORMAT)

def get_now():
    return datetime.datetime.now().isoformat()

def run_with_timeout_retry(chain, chain_input, timeout=TIMEOUT):
    chain.llm.request_timeout = timeout
    for i in range(TIMEOUT_RETRY):
        try:
            return chain(chain_input)
        except OpenAITimeoutError as e:
            print("Retrying...")
            time.sleep(1)
    raise Exception("Timeout")

def loadConversationOneDay(conversation, ai_prefix, human_prefix, add_date=True):
    # Force add date to False to avoid retruning the timestamp on the output
    # TODO: Might reconsider again when want LLM to be time-sensitive.
    add_date = False

    out = []
    for chat in conversation["conversation"]:
        for k, v in chat["content"].items():
            if k.lower() != ai_prefix.lower() and k.lower() != human_prefix.lower():
                continue

            if add_date:
                content = "{} ({})".format(v, chat["time"],)
            else:
                content = v
            currentDict = {
                "type": k,
                "data": {
                    "content": content,
                    "additional_kwargs": {}
                }
            }
            out.append(currentDict)
    
    return out

def conversation_to_string(conversation, to_string=False, add_date=True):
    """ Convert the converstation dicct list to covnersation string list
    """
    # Force add date to False to avoid retruning the timestamp on the output
    # TODO: Might reconsider again when want LLM to be time-sensitive.
    add_date = False

    if to_string:
        out = ""
    else:
        out = []

    for line in conversation["conversation"]:
        for k, v in line["content"].items():
            if add_date:
                s = "{}: {} ({})\n".format(k, v, line["time"])
            else:
                s = "{}: {}\n".format(k, v)

            if to_string:
                out += s
            else:
                out.append(s)

    return out

def loadAllConversationsToMemory(conversations, ai_prefix, human_prefix):
    out = []
    for sessions in conversations:
        out.extend(loadConversationOneDay(sessions, ai_prefix, human_prefix))

    retrieved_messages = messages_from_dict(out)
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    return retrieved_chat_history

def progressive_summarise(chain, prev_info, follow_up_conv):
    print("Progressively summarising")
    for conv in follow_up_conv:
        conv = "\n".join(conv)
        conversation_info = run_with_timeout_retry(chain, {"conversation": conv,
                                                            "summary": prev_info})["text"]
        prev_info = conversation_info

    return conversation_info

def chunk_conversation(conversation):
    follow_up_conv = []

    if len(conversation) > MAX_CONV_LENGTH:
        print("Converseation is too long, chunking it", len(conversation), MAX_CONV_LENGTH)
        # In the conversation is too long, progressively summarise the key information
        num_chunks = math.ceil(len(conversation) / MAX_CONV_LENGTH)
        
        for i in range(1, num_chunks):
            current_conv = conversation[i*MAX_CONV_LENGTH:(i+1)*MAX_CONV_LENGTH]
            follow_up_conv.append(current_conv)
    
        conversation = conversation[:MAX_CONV_LENGTH]

    return conversation, follow_up_conv
    

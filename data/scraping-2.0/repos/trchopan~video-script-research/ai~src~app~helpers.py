import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, PromptValue


def find_element(array, condition):
    for element in array:
        if condition(element):
            return element
    return None


def get_timestamp():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)


def send_chat_prompt_with_print(chat: ChatOpenAI, prompt: PromptValue) -> BaseMessage:
    print(">>>>")
    for m in prompt.to_messages():
        print(m.content)

    result = chat(prompt.to_messages())
    print(result.content)
    print("===")
    return result


def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

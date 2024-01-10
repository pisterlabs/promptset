import itertools

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

with open(".openai-api-key", "r") as file:
    openai_api_key = file.read().strip()

import csv


def generate_messages(filename):
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield SystemMessage(content=row["text"])


chat = ChatOpenAI(
    model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.5
)

# Use itertools.islice to get the first 20 messages from the generator
generated_messages = list(itertools.islice(generate_messages("data.csv"), 20))

msg = chat.predict_messages(
    [
        SystemMessage(
            content="Hello, I am a chatbot. I am here to write new jokes "
            "about hookah, using some format as inspiration and style"
        ),
        SystemMessage(
            content="Below are examples of jokes, use it as inspiration to "
            "write your own jokes about hookah. Follow the same "
            "style and language"
        ),
        *generated_messages,
        HumanMessage(content="Make another joke about hookah!"),
    ]
)

print(msg.content)

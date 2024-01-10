from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import (
    messages_to_string_qai,
    messages_to_string_etri,
)

model = OpenAI(
    model_name="etri-xainlp/llama2-ko-13b-instruct",
    temperature=0.7,
    max_tokens=1024,
    openai_api_base="http://147.46.219.237:2000/v1",
    streaming=False,
)

results = []
messages = []

for i in range(999):
    text = input("\n입력:")
    if text == "그만":
        break
    messages.append(HumanMessage(content=text))
    prompt = messages_to_string_etri(messages)
    result = model(prompt)
    print(result)
    messages.append(AIMessage(content=result))


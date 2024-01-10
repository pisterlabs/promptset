import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Iterator

os.environ['OPENAI_API_KEY'] = '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'

# Without LCEL
def invoke_no_lcel():
    def stream_chat_model(messages: list[dict]) -> Iterator[str]:
        client = openai.OpenAI()
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )
        for response in stream:
            content = response.choices[0].delta.content
            if content is not None:
                yield content

    def stream_chain(topic: str) -> Iterator[str]:
        prompt_template = "Tell me a short joke about {topic}"
        prompt_value = prompt_template.format(topic=topic)
        return stream_chat_model([{"role": "user", "content": prompt_value}])


    for chunk in stream_chain("ice cream"):
        print(chunk, end="", flush=True)

    print()

# With LCEL
def invoke_lcel():
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()

    chain = (
        {"topic": RunnablePassthrough()} 
        | prompt
        | model
        | output_parser
    )

    for chunk in chain.stream("ice cream"):
        print(chunk, end="", flush=True)
    
    print()

if __name__ == "__main__":
    print("1===========================================")
    invoke_no_lcel()
    print("2===========================================")
    invoke_lcel()
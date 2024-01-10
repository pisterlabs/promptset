import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ['OPENAI_API_KEY'] = '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'

# Without LCEL
def invoke_no_lcel():
    prompt_template = "Tell me a short joke about {topic}"
    client = openai.OpenAI()


    def call_chat_model(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
        )
        return response.choices[0].message.content

    def invoke_chain(topic: str) -> str:
        prompt_value = prompt_template.format(topic=topic)
        messages = [{"role": "user", "content": prompt_value}]
        return call_chat_model(messages)

    print(invoke_chain("ice cream"))

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

    print(chain.invoke("ice cream"))


if __name__ == "__main__":
    print("1===========================================")
    invoke_no_lcel()
    print("2===========================================")
    invoke_lcel()
import os
import openai
import asyncio


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ['OPENAI_API_KEY'] = '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'



# Without LCEL
async def invoke_no_lcel():
    prompt_template = "Tell me a short joke about {topic}"
    client = openai.OpenAI()
    async_client = openai.AsyncOpenAI()

    async def acall_chat_model(messages: list[dict]) -> str:
        response = await async_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
        )
        return response.choices[0].message.content

    async def ainvoke_chain(topic: str) -> str:
        prompt_value = prompt_template.format(topic=topic)
        messages = [{"role": "user", "content": prompt_value}]
        return await acall_chat_model(messages)
    
    print(await ainvoke_chain("ice cream"))

# With LCEL
async def invoke_lcel():
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()

    chain = (
        {"topic": RunnablePassthrough()} 
        | prompt
        | model
        | output_parser
    )

    print(await chain.ainvoke("ice cream"))

if __name__ == "__main__":
    print("1===========================================")
    #asyncio.run(invoke_no_lcel())
    print("2===========================================")
    asyncio.run(invoke_lcel())
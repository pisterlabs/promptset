from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import asyncio
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
prompt = PromptTemplate.from_template("please tell a joke about {role}")
chain = prompt | model

#this is how we "batch" input into an llm
response1 = chain.batch([{"role":"a software programmer"}, {"role":"a marriage planner"}])

#this is how we stream the answer
for s in chain.stream({"role":"a software programmer"}):
        print(s.content, end="")

#this is how we use async
# the idea is, if main is waiting for the API call, let's call func2 so we don't waste time, we need 2 functions to demo async
# this is not parallelism, it still runs one task at a time, but it can jump to run different tasks from different functions.
async def main(role):
    asyncio.create_task(func2())
    print("let's start")
    response = await chain.ainvoke({"role":role})
    print(f" this is the {response.content}")

async def func2():
    print(f"Yo, don't waste time, let's print something")
    await asyncio.sleep(3)
    print("how about me?")

asyncio.run(main("a movie producer"))

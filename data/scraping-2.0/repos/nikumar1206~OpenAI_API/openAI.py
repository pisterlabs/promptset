import asyncio
import sys

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import config


async def chatbot() -> None:
    print("Hello! I'm a chatbot powered by ChatGPT. How can I help you today?")
    memory = ConversationBufferMemory(ai_prefix="Assistant")
    prompt_template = PromptTemplate(
        template="""You are a helpful assistant that helps with any and all questions to the best of your ability.You may be given some chat history from a previous conversation.
        {history}
        Human:{input}
        Assistant:""",
        input_variables=["history", "input"],
    )
    llm_chain = LLMChain(
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=config.API_KEY,
            client="hello",
            model="gpt-3.5-turbo",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
        prompt=prompt_template,
        memory=memory,
    )

    while True:
        await llm_chain.arun(input=input("\nYou: "))


if __name__ == "__main__":
    try:
        asyncio.run(chatbot())
    except (KeyboardInterrupt, EOFError):
        print("Goodbye!")
        sys.exit()
    except Exception as err:
        sys.exit(f"Error has occured {err=}")

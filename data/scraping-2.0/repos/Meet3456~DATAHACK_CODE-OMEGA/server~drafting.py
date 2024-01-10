#8088
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import chainlit as cl
import os
from dotenv import load_dotenv
import openai
import chainlit as cl

welcome_message = "NyayMitra: Your go-to for demystifying law! Easily Draft Different Documents with Ease!! Ask now!! 🌐⚖️ #LegalInsights #NyayMitra"

api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

model_name = "ft:gpt-3.5-turbo-0613:codeomega::8Ep5ku4U"
# settings = {

#     "top_p": 1,
#     "frequency_penalty": 0,
#     "presence_penalty": 0,
# }


@cl.on_chat_start
async def start_chat():
    await cl.Message(content=welcome_message).send()
    cl.user_session.set(
         
        "message_history",
         [
            {"role": "system", "content": "LawYantra is a factual chatbot which generates   the complete legal document according to the user query."},
            {"role": "user", "content": "Forge a domicile rental arrangement"}
         ]   
)


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    async for stream_resp in await openai.ChatCompletion.acreate(
        model=model_name, messages=message_history, stream=True
    ):
        token = stream_resp.choices[0]["delta"].get("content", "")
        await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.send()




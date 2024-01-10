#Assists the user with interpreting and replying to parent, admin, and coworker emails.
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMChain, ConversationChain, SequentialChain
from langchain.chat_models import ChatOpenAI, ChatGooglePalm

gpt4 = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    frequency_penalty=0.5,
    streaming=True,
)

email_system_msg = "You are an empathetic A




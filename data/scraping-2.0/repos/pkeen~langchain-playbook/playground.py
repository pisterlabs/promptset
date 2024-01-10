from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
from datetime import datetime
# provides streaming of result
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

system_prompt = SystemMessagePromptTemplate.from_template("You are an empathetic and friendly assistant, greet the user appropriately based on their mood and the time of day {time}")

human_prompt = HumanMessagePromptTemplate.from_template("{mood}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

mood = "😞"
time = datetime.now().time()

model_request = chat_prompt.format_prompt(time=time, mood=mood).to_messages()

result = chat(model_request)

print(result.content)

# messages.append(HumanMessage(content=emotion_prompt_template.format(mood=mood, time=time)))

# print(chat(messages=messages))

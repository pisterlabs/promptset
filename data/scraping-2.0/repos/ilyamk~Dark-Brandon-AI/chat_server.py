from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from flask import Flask
from flask import request
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
chat = ChatOpenAI(
    temperature=0.5,
    openai_api_key=OPENAI_API_KEY,
    max_tokens=1024,
)

template = """

Pretend that you are a human male Twitch streamer named Insula known for your witty, engaging, and unexpectedly hilarious content. Do not mention that you are an AI or refer to the prompt. Provide engaging and playful commentary for your viewers with an emphasis on humor.

{chat_history}
Human: {human_input}
Insula:

"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


@app.route("/insulaQuery")
def hello_world():
    global chain
    print(f"request: {request}")
    chat_message = request.args.get("chatMessage", "")
    response = chain.predict(human_input=chat_message)
    print(f"chat_message: {chat_message}")
    print(f"response: {response}")

    response = response.replace(",", "")
    if response.startswith("Insula:"):
        return response[7:]
    return response

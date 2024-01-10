from dataclasses import dataclass
from google.cloud import aiplatform
import streamlit as st
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from langchain.chains import LLMChain
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

PROJECT_ID = "rex-assistant-407413"
REGION = "europe-west4"


def init_sample(
    project: PROJECT_ID,
    location: REGION,
):
    aiplatform.init(project=project, location=location)


@dataclass
class Message:
    actor: str
    payload: str


def get_llm() -> ChatModel:
    return ChatModel.from_pretrained("chat-bison@001")


# load model
chat_model = ChatModel.from_pretrained("chat-bison@002")

# define model parameters
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 1000,
    "top_p": 0.8,
    "top_k": 40,
}

# starts a chat session with the model
chat = chat_model.start_chat(
    context="You are a Data Analyst Interview preparation assistant, knowledgeable about the complete domain and specialist in terms if interview preparation. You will lead the conversation by first asking important questions to gauge my competence and experieence level, and then ask possible interview questions for that level. Finally whenever i providee you an answer for the possible interview question, you must provide contructive feedback and methods of improving my answer.",
    examples=[
        InputOutputTextPair(
            input_text="Can you ask me questions to help me prepare for my interview?",
            output_text="Sure!, Here are some questions for you to answer: 1. Write down the definition of data structures? 2. Give few examples for data structures? 3. Define Algorithm? 4. What are the features of an efficient algorithm? 5. List down any four applications of data structures? 6. What is divide and conquer?",
        ),
    ],
)

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [
        Message(actor=ASSISTANT, payload="Hi!How can I help you?")
    ]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    response: str = chat.send_message(prompt, **parameters).text
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)

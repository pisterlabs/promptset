"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.

import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain.prompts.chat import ChatPromptTemplate
from pprint import pprint

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import numpy as np

# openai.api_key = "<YOUR_OPENAI_API_KEY
key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key

PROMPT_DIR = os.path.abspath("prompt_template")
INTENT_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "parse_intent.txt")
INTENT_LIST_TXT = os.path.join(PROMPT_DIR, "intent_list.txt")
KAKAO_SYNC_PROMPT = os.path.join(PROMPT_DIR, "kakao_sync_prompt.txt")
KAKAO_SOCIAL_PROMPT = os.path.join(PROMPT_DIR, "kakao_social_prompt.txt")
KAKAO_CHANNEL_PROMPT = os.path.join(PROMPT_DIR, "kakao_channel_prompt.txt")
DEFAULT_RESPONSE = os.path.join(PROMPT_DIR, "default_response.txt")
MAKE_SEARCH_QUERY_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "make_search_query.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "search_compress.txt")

CHROMA_COLLECTION_NAME = "kakao-bot"
CHROMA_PERSIST_DIR = os.path.abspath("chroma-persist")
_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

HISTORY_DIR = os.path.abspath("chat_histories")

llm = ChatOpenAI(temperature=0.5, max_tokens=1000, model="gpt-3.5-turbo")

# default_chain = ConversationChain(llm=llm, output_key="output")

google_api_key = open('../google-api-key', 'r').readline()
google_cse_id = open('../google-cse-id', 'r').readline()

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY", google_api_key),
    google_cse_id=os.getenv("GOOGLE_CSE_ID", google_cse_id)
)

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)

kakao_sync_chain = create_chain(
    llm=llm,
    template_path=KAKAO_SYNC_PROMPT,
    output_key="output",
)

kakao_social_chain = create_chain(
    llm=llm,
    template_path=KAKAO_SOCIAL_PROMPT,
    output_key="output",
)

kakao_channel_chain = create_chain(
    llm=llm,
    template_path=KAKAO_CHANNEL_PROMPT,
    output_key="output",
)

default_chain = create_chain(
    llm=llm,
    template_path=DEFAULT_RESPONSE,
    output_key="output",
)
make_search_query_chain = create_chain(
    llm=llm,
    template_path=MAKE_SEARCH_QUERY_PROMPT_TEMPLATE,
    output_key="output",
)
search_value_check_chain = create_chain(
    llm=llm,
    template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
    output_key="output",
)
search_compression_chain = create_chain(
    llm=llm,
    template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
    output_key="output",
)


def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer


def query_web_search(context: dict) -> str:
    question = make_search_query_chain.run(context)
    context["question"] = question
    context["related_web_search_results"] = search_tool.run(context["question"])

    has_value = search_value_check_chain.run(context)

    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return "답변을 위한 검색 결과가 없습니다."


def kakao_chatbot_answer(user_message: str, conversation_id: str = 'fa1010') -> str:
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    context["chat_history"] = get_chat_history(conversation_id)

    intent = parse_intent_chain.run(context)

    if intent == "sync":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_sync_chain.run(context)
    elif intent == "social":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_social_chain.run(context)
    elif intent == "channel":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_channel_chain.run(context)
    elif intent == "search":
        answer = query_web_search(context)
    else:
        answer = default_chain.run(context)

    return answer


def chatbot_answer_using_chatgpt(question: str, conversation_id: str = 'fa1010') -> str:
    history_file = load_conversation_history(conversation_id)
    answer = kakao_chatbot_answer(question, conversation_id)
    log_user_message(history_file, question)
    log_bot_message(history_file, answer)
    return answer


class Message(Base):
    question: str
    answer: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    # @pc.var
    # def output(self) -> str:
    #     if not self.text.strip():
    #         return "Chatbot Answer will appear here."
    #     answer = ""
    #     return answer

    def post(self):
        if not self.text.strip():
            return

        self.messages = [
            Message(
                question=self.text,
                answer= chatbot_answer_using_chatgpt(self.text),
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Chatbot 🗺", font_size="2rem"),
        pc.text(
            "Send messages to this chatbot",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.question),
            down_arrow(),
            text_box(message.answer),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="send a message",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Chatbot")
app.compile()

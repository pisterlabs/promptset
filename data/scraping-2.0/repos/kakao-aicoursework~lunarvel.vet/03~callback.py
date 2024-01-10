from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging

import openai
import os
from os import environ

# from langchain                  import LLMChain
from langchain.chains           import LLMChain
from langchain.chat_models      import ChatOpenAI
from langchain.text_splitter    import CharacterTextSplitter
from langchain.prompts.chat     import HumanMessagePromptTemplate
from langchain.prompts.chat     import ChatPromptTemplate
from langchain.schema           import SystemMessage
from langchain.memory           import ConversationBufferMemory
from langchain.memory           import FileChatMessageHistory
from langchain.tools            import Tool
from langchain.utilities        import GoogleSearchAPIWrapper
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

openai.api_key = environ.get("API_KEY")

CUR_DIR = os.getcwd()
# BUG_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/bug_analyze.txt")
# BUG_STEP2_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/bug_solution.txt")
# ENHANCE_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/enhancement_say_thanks.txt")

KAKAO_CHANNEL_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/bug_analyze.txt")
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/default_response.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/parse_intent.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "data/intent_list.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/search_compress.txt")
HISTORY_DIR = os.path.join(CUR_DIR, "chat_histories")
CHROMA_DIR = os.path.join(CUR_DIR, "chromadb")


intent_list_content = ["""
kakao_sync: Questions about kakao sync. This service syncs messages, photos, and videos from KakaoTalk between your mobile and PC.
kakao_talkchannel: Questions about kakao channel. It's a brand marketing platform that facilitates communication between businesses and users.
kakao_social: Questions about kakao social. This is kakao's social media service where users can share information and communicate with each other.
""", """
kakao_sync: 카카오 싱크에 대한 질문. 이 서비스는 모바일과 PC 간의 카카오톡 메시지, 사진, 동영상 등을 동기화해주는 서비스입니다.
kakao_talkchannel: 카카오 채널에 대한 질문. 기업과 사용자 사이의 원활한 소통을 위해 제공되는 브랜드 마케팅 플랫폼입니다. 
kakao_social: 카카오 소셜에 대한 질문. 사용자들이 정보를 공유하고 소통할 수 있는 카카오의 소셜 미디어 서비스.
""" ]

parse_intent_content = """
Your job is to select one intent from the <intent_list>.

<intent_list>
{intent_list}
</intent_list>

User: {user_message}
Intent:
"""


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_prompt_template(path: str) -> str:
    return read_file(path)



def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_file(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

from enum import Enum

g_intent_str = ["kakao_social", "kakao_sync", "kakao_talkchannel"]
class Intent:
    NONE                = -1
    kakao_social        = 0
    kakao_sync          = 1
    kakao_talkchannel   = 2
    str = ["kakao_social", "kakao_sync", "kakao_talkchannel"]

    def to_str(self) -> str:
        return self.str[self.value]

    def to_idx(self) -> int:
        return self.value

    def __init__(self, value: int = -1):
        self.value = value

    @classmethod
    def init(cls, value_str: str):
        value_str = value_str.lower()
        ret = cls(cls.NONE)

        if value_str == 'kakao_social':
            ret = cls(cls.kakao_social)
        elif value_str == 'kakao_sync':
            ret = cls(cls.kakao_sync)
        elif value_str == 'kakao_talkchannel':
            ret = cls(cls.kakao_talkchannel)

        return ret

class ConversationHistory:
    def __init__(self, conversation_id: str):
        file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")

        self.conversation_id = conversation_id
        self.history = FileChatMessageHistory(file_path)

    @classmethod
    def load(cls, conversation_id: str):
        return ConversationHistory(conversation_id)

    def save_history(self, user_message, bot_answer):
        self.log_user_message(user_message)
        self.log_bot_message(bot_answer)

    def log_user_message(self, user_message: str):
        self.history.add_user_message(user_message)

    def log_bot_message(self, bot_message: str):
        self.history.add_ai_message(bot_message)

    def get_chat_history(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            chat_memory=self.history,
        )
        return memory.buffer

from chromadb.utils import embedding_functions

class IntentDB:
    intent = Intent(Intent.NONE)
    path: str               = None
    # db: chromadb.Collection = None
    # retriever = None

    def __init__(self, client: chromadb.PersistentClient, intent: Intent, txt_path: str):
        self.db: Chroma = None
        self.intent = intent
        self.conversation_history = ConversationHistory(intent.to_str())

        self.init_db(intent, txt_path)


    def query(self, query: str, use_retriever: bool = False) -> list[str]:
        if use_retriever:
            retriever = self.db.as_retriever()
            docs = retriever.get_relevant_documents(query)
        else:
            docs = self.db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs

    # def add(self, documents, ids):
        # self.db.add(documents=documents, ids=ids)

    def init_db(self, intent: Intent, txt_path: str):
        # read from txt_path into raw_text
        self.intent = intent
        self.path = txt_path

        raw_text = read_file(txt_path)

        # todo: 전처리가 많이 필요하다.
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_text(raw_text)

        text_dict = {"id": [], "text": texts}

        # todo: id 도 의미를 가질수 있도록...
        text_dict["id"] = [f"{i}" for i in range(len(texts))]

        # create the open-source embedding function
        # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(CHROMA_DIR) == False:
            # 없으면, 생성
            self.db = Chroma.from_documents(
                documents=text_dict["text"],
                ids=text_dict["id"],
                embedding_function=OpenAIEmbeddings(),
                persist_directory=CHROMA_DIR,
                collection_name=self.intent.to_str(),
                # metadata={"hnsw:space": "cosine"},
            )
        else:
            # load docs into Chroma DB
            self.db = Chroma(
                persist_directory=CHROMA_DIR,
                collection_name=self.intent.to_str(),
                embedding_function=OpenAIEmbeddings(),
                # metadata={"hnsw:space": "cosine"},
            )
            self.db.get()

        return
from langchain.utilities import DuckDuckGoSearchAPIWrapper

def create_search_tool(use_google: bool = False):
    search = None
    name = ""
    description = ""
    search = None

    if use_google == False:
        # use DuckDuckGo
        search = DuckDuckGoSearchAPIWrapper(region="ko-KR")
        name = "DuckDuckGo Search"
    else:
        search = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            google_cse_id=os.getenv("GOOGLE_CSE_ID", "")
        )
        name = "Google Search"

    # todo: 명확하게 주어주면, 성능개선에 도옴이 된다.
    description = f"Search {name} for recent results."
    search_tool = Tool(
        name=name,
        description=description,
        func=search.run
    )

    return search_tool

import os
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    context["related_web_search_results"] = gChatBot.search_tool.run(user_message)

    has_value = gChatBot.search_value_check_chain.run(context)

    print(has_value)
    if has_value == "Y":
        return gChatBot.search_compression_chain.run(context)
    else:
        return ""


def intended_query(intent: Intent, query: str) -> list[str]:
    context = dict(
        related_documents=[],
        user_message=query
    )

    i = intent.to_idx()

    if i == Intent.NONE:
        # web search with query

        # todo: 모든 DB를 다 봐야할까?
        for i in range(len(gChatBot.intent_info)):
            info = gChatBot.intent_info[i]
            db = info['db']
            context["related_documents"].append(db.query(context["user_message"]))

        context["compressed_web_search_results"] = query_web_search(context["user_message"])
        answer = gChatBot.default_chain.run(context)
    else:
        info = gChatBot.intent_info[i]
        db = info['db']
        chain = info['chatbot_chain']

        context["related_documents"] = db.query(context["user_message"])
        answer = chain.run(context)

    return answer

def gernerate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
    hist = ConversationHistory.load(conversation_id)

    context = dict( user_message=user_message )
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    context["chat_history"] = hist.get_chat_history()

    # intent = parse_intent_chain(context)["intent"]
    intent_str = gChatBot.parse_intent_chain.run(context)
    print(intent_str)
    print("======================")


    intent = Intent.init(intent_str)

    answer = intended_query(intent, user_message)

    # save history
    hist.save_history(user_message, answer)

    return {'answer': answer}

class ChatBot:
    llm: ChatOpenAI = None

    client: chromadb.PersistentClient = None

    parse_intent_chain: LLMChain = None

    # db: Datastore = []
    # chatbot_chain: LLMChain = []
    intent_info = []

    search_tool: Tool = None
    default_chain: LLMChain = None

    search_value_check_chain: LLMChain = None
    search_compression_chain: LLMChain = None

    def init(self):
        if os.path.exists(HISTORY_DIR) == False:
            os.makedirs(HISTORY_DIR)

        os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

        self.llm = llm = ChatOpenAI(
            temperature=0.1,
            max_tokens=200,
            model="gpt-3.5-turbo"
        )

        # init DBs
        pwd = os.getcwd()
        self.client = chromadb.PersistentClient(pwd + "/chromadb")

        # init intent_info
        for i in range(len(Intent.str)):
            intent_str = Intent.str[i]
            prompt_template = os.path.join(CUR_DIR, f'data/prompt_template_{intent_str}.txt')

            _tmp_ = {
                'db': IntentDB(self.client, Intent(i), f'data/project_data_{intent_str}.txt'),
                'chatbot_chain': create_chain(
                    llm=llm,
                    template_path=prompt_template,
                    output_key="output_" + intent_str
                )
            }

            self.intent_info.append(_tmp_)

            # self.chatbot_chain[i] = create_chain(
            #     llm=llm,
            #     template_path=prompt_template,
            #     output_key="output_"+intent_str,
            # )

        self.search_tool = create_search_tool()

        self.parse_intent_chain = create_chain(
            llm=llm,
            template_path=INTENT_PROMPT_TEMPLATE,
            output_key="intent",
        )

        self.default_chain = create_chain(
            llm=llm,
            template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE,
            output_key="output"
        )

        self.search_value_check_chain = create_chain(
            llm=llm,
            template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
            output_key="output",
        )

        self.search_compression_chain = create_chain(
            llm=llm,
            template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
            output_key="output",
        )

gChatBot = ChatBot()
gChatBot.init()

async def callback_handler(request: ChatbotRequest) -> dict:
    # raw_data = read_file("data/project_data_kakaosync.txt")
    #
    # system_message = "assistant는 챗봇으로 동작한다. 챗봇은 '제품정보' 내용을 참고하여, user의 질문 혹은 요청에 따라 적절한 답변을 제공합니다."
    # human_template = ("제품정보: {product_data}\n" +
    #                   request.userRequest.utterance )
    #
    # system_message_prompt = SystemMessage(content=system_message)
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    #
    # chat = ChatOpenAI(temperature=0.8)
    #
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    #
    # chain = LLMChain(llm=chat, prompt=chat_prompt)
    # output_text = chain.run(product_data=raw_data)

    ret = gernerate_answer(request.userRequest.utterance)
    output_text = ret['answer']

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text[0] + "\n" + output_text[1]
                    }
                }
            ]
        }
    }

    # debug
    print(output_text[0] + "\n" + output_text[1])

    time.sleep(1.0)

    url = request.userRequest.callbackUrl


    print(output_text)

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload) as resp:
                await resp.json()


# async def callback_handler1(request: ChatbotRequest) -> dict:
#     os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
#
#     raw_data = read_file("data/project_data_kakaosync.txt")
#
#     system_message = "assistant는 챗봇으로 동작한다. 챗봇은 '제품정보' 내용을 참고하여, user의 질문 혹은 요청에 따라 적절한 답변을 제공합니다."
#     human_template = ("제품정보: {product_data}\n" +
#                       request.userRequest.utterance )
#
#     system_message_prompt = SystemMessage(content=system_message)
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#
#     chat = ChatOpenAI(temperature=0.8)
#
#     chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#
#     chain = LLMChain(llm=chat, prompt=chat_prompt)
#
#     output_text = chain.run(product_data=raw_data)
#
#     payload = {
#         "version": "2.0",
#         "template": {
#             "outputs": [
#                 {
#                     "simpleText": {
#                         "text": output_text
#                     }
#                 }
#             ]
#         }
#     }
#
#     # debug
#     print(output_text)
#
#     time.sleep(1.0)
#
#     url = request.userRequest.callbackUrl
#
#     if url:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url=url, json=payload) as resp:
#                 await resp.json()


# async def callback_handler2(request: ChatbotRequest) -> dict:
#
#     # ===================== start =================================
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": SYSTEM_MSG},
#             {"role": "user", "content": request.userRequest.utterance},
#         ],
#         temperature=0,
#     )
#     # focus
#     output_text = response.choices[0].message.content
#
#    # 참고링크 통해 payload 구조 확인 가능
#     payload = {
#         "version": "2.0",
#         "template": {
#             "outputs": [
#                 {
#                     "simpleText": {
#                         "text": output_text
#                     }
#                 }
#             ]
#         }
#     }
#     # ===================== end =================================
#     # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
#     # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format
#
#     time.sleep(1.0)
#
#     url = request.userRequest.callbackUrl
#
#     if url:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url=url, json=payload, ssl=False) as resp:
#                 await resp.json()
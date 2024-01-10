import os
import openai
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from database import DataBase
from history import History

openai.api_key = os.environ["GPT_KEY"]
os.environ["OPENAI_API_KEY"] = os.environ["GPT_KEY"]

CUR_DIR = os.path.dirname(os.path.abspath('./kakaochattest_guide'))
DB_SEARCH_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/db_search_response.txt")
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/default_response.txt")
NORMAL_RESPONSE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/normal_response.txt")
TRANSLATE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/translate.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/parse_intent.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt/search_compress.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "prompt/intent_list.txt")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


class LLMGenerator:
    def __init__(self):
        self.db = DataBase(CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
        self.search = LLMGenerator.initialize_search_api()
        self.history = History()
        self.llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")
        self.initialize_chains(self.llm)
        self.generate_tools()

    def generate_tools(self):
        self.search_tool = Tool(
            name="Google Search",
            description="Search Google for recent results.",
            func=self.search.run,
        )

    def query_web_search(self, user_message: str) -> str:
        translation = self.chains["translate_chain"].run({"user_message": user_message})
        context = {"user_message": user_message, "related_web_search_results": self.search_tool.run(translation)}
        has_value = self.chains["search_value_check_chain"].run(context)
        if has_value == "Y":
            return self.chains["search_compression_chain"].run(context)
        else:
            return ""

    def initialize_chains(self, llm):
        self.chains = {}
        self.chains["싱크_db_search_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=DB_SEARCH_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["소셜_db_search_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=DB_SEARCH_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["톡채널_db_search_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=DB_SEARCH_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["parse_intent_chain"] = LLMGenerator.create_chain(
            llm=llm,
            template_path=INTENT_PROMPT_TEMPLATE,
            output_key="intent",
        )
        self.chains["default_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["normal_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=NORMAL_RESPONSE_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["translate_chain"] = LLMGenerator.create_chain(
            llm=llm, template_path=TRANSLATE_PROMPT_TEMPLATE, output_key="output"
        )
        self.chains["search_value_check_chain"] = LLMGenerator.create_chain(
            llm=llm,
            template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
            output_key="output",
        )
        self.chains["search_compression_chain"] = LLMGenerator.create_chain(
            llm=llm,
            template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
            output_key="output",
        )

    def request_query(self, user_message: str) -> dict:

        context = dict(user_message=user_message)
        context["input"] = context["user_message"]
        context["intent_list"] = LLMGenerator.read_prompt_template(INTENT_LIST_TXT)
        context["chat_history"] = self.history.get_chat_history()

        intent = self.chains["parse_intent_chain"].run(context)
        print(intent)
        if "카카오싱크" in intent:
            context["topic"] = "카카오싱크"
            context["related_documents"] = self.db.query_db(context["user_message"])
            answer = self.chains["싱크_db_search_chain"].run(context)
        elif "카카오소셜" in intent:
            context["topic"] = "카카오소셜"
            context["related_documents"] = self.db.query_db(context["user_message"])
            answer = self.chains["소셜_db_search_chain"].run(context)
        elif "카카오톡채널" in intent:
            context["topic"] = "카카오톡채널"
            context["related_documents"] = self.db.query_db(context["user_message"])
            answer = self.chains["톡채널_db_search_chain"].run(context)
        elif "심화질문" in intent:
            context["topic"] = "검색 결과에 심화 질문"
            context["related_documents"] = self.db.query_db(context["user_message"])
            context["compressed_web_search_results"] = self.query_web_search(
                context["user_message"]
            )
            answer = self.chains["default_chain"].run(context)
        else:
            answer = self.chains["normal_chain"].run(context)

        self.history.log_user_message(user_message)
        self.history.log_bot_message(answer)
        return {"answer": answer}

    @staticmethod
    def read_prompt_template(file_path: str) -> str:
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template

    @staticmethod
    def create_chain(llm, template_path, output_key):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=LLMGenerator.read_prompt_template(template_path)
            ),
            output_key=output_key,
            verbose=True,
        )

    @staticmethod
    def initialize_search_api():
        search = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyAvj4AijLuwucptaXj5Lw3L0LLExeythiY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID", "267c48091a8f34a8e")
        )

        return search

if __name__ == "__main__":
    generator = LLMGenerator()
    # print(generator.request_query("카카오싱크 실행 방법은?"))
    result = generator.search_tool.run("Obama's first name?")
    print(result)
from langchain.agents import load_tools
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
os.environ["OPENAI_KEY"] = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_KEY")
wolf = os.getenv("WOLFRAM_ALPHA")




os.environ["WOLFRAM_ALPHA_APPID"] = os.getenv("WOLFRAM_ALPHA")
class LangService:
    def __init__(self):
        pass


    def model(self,query: str):

        # llm = OpenAI(temperature=0)

        # tools = load_tools(["serpapi","llm-math","wolfram-alpha"],llm=llm,serpapi_api_key=serpapi,
        #                                                                wolfram_alpha_appid=wolf)

        # agent = initialize_agent(tools,llm, agent="zero-shot-react-description",verbose=True)

        # return agent.run(query,length=256) 


        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        search = SerpAPIWrapper()
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        wolfram = WolframAlphaAPIWrapper()
        tools = [
            Tool(
                name = "Search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions"
            ),
            Tool(
                name="ChatGPT",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions that can answer ChatGPT"
            ),
            Tool(
                name="Wolf",
                func=wolfram.run,
                description="useful for when you need to answer questions about math"
            )
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


        return agent.run(query)

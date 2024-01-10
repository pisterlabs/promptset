from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import os
#from agent_rag import agent_rag_executor
#from retriever import faq_db

# math tool
llm_math = LLMMathChain.from_llm(OpenAI(api_key=os.environ["OPENAI_API_KEY"],))
llm_math_tool = Tool(
        name="Calculator",
        func=llm_math.run,
        description="useful for when you need to answer questions about math"
    )

# custom search tool
class CustomDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool with a custom query prefix."""
        # Füge den Präfix zur Suchanfrage hinzu
        modified_query = f"site:www.eak.admin.ch {query}"
        # Rufe die _run-Methode der Basisklasse mit der modifizierten Anfrage auf
        return super()._run(modified_query, run_manager)
wrapper = DuckDuckGoSearchAPIWrapper(region="ch-de", time="y", max_results=1)
custom_search_tool = CustomDuckDuckGoSearchRun(
    name="EAK-Search",
    description="Searches the www.eak.admin.ch website from Eidg. Ausgleichskasse EAK, for your query",
    api_wrapper=wrapper)

# word length tool
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# summary chain
template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""
prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)
summary_chain = LLMChain(
    llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    prompt=prompt,
    verbose=True,
    memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
)
summary_tool = Tool (
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    )

# faq rag
# Create a retriever tool
# retriever = faq_db.as_retriever(
#         search_type="mmr",
#         search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.85}
#         )
# # Create a tool that uses the retriever
# faq_mmr_retriever = create_retriever_tool(
#     retriever,
#     "sozialversicherungssystem_faq_retriever",
#     "Searches and returns faq documents about swiss social security system in german, french and italian.",
# )

#########
# tools #
#########
tools = [
    #faq_mmr_retriever,
    summary_tool,
    get_word_length,
    custom_search_tool,
    llm_math_tool]

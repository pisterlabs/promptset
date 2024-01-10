from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents import AgentExecutor
from tqdm import tqdm

local_db_path = "brenda_chroma_db"
# embeddings = OpenAIEmbeddings() # too slow 
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=local_db_path, embedding_function=embedding_function)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

tool = create_retriever_tool(
    retriever,
    "search_your_document",
    "Searches and returns documents from your document."
)
tools = [tool]

# Create the conversational retrieval agent
llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")

# set up prompt
memory_key = "history"

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

system_message = SystemMessage(
    content=(
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# Interact with the agent
if __name__ == "__main__":
    result = agent_executor({"input": "Tell me about the PET hydrolase enzyme."})
    print(result["output"])

# result = agent_executor({"input": "What did you do?"})
# print(result["output"])
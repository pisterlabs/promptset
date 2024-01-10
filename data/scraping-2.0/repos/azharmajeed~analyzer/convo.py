import time
from langchain.agents import (
    Tool,
    AgentType,
    AgentExecutor,
    LLMSingleActionAgent,
)
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.document_loaders import PyPDFLoader
import os
import pinecone
from typing import List, Tuple
from dotenv import load_dotenv

from prompt_template import CustomOutputParser, CustomPromptTemplate


class Conversation:
    def __init__(self, conv_list: List[Tuple]):
        # user_session = None
        # self.conv_id
        # self.conv_created_datetime
        # self.conv_last_updated_datetime
        # self.conv_type
        # self.conv_meta
        # self.conv_mem
        # self.docs
        # self.images
        # self.conv_dsources
        self.conv_token_list = conv_list

    def tools_bag(self, penal_code):
        search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="Current Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world",
            ),
            Tool(
                name="Penal Code QA System",
                func=penal_code.run,
                description="useful for when you need to answer questions about the Indian Penal Code or if any issue is associated with it.",
            ),
            # Tool(
            #     name="Conversation History chain",
            #     func=conversation_chain.run,
            #     description="useful when you want to refer to entities and their related information from the previous conversation history",
            # ),
        ]
        return tools

    def vector_store(self, docs, index_name, embeddings, locol_vectorstore):
        if locol_vectorstore:
            docsearch = Chroma.from_documents(
                docs, embeddings, collection_name=index_name
            )
        else:
            memory_index_name = "chat-memory-index"
            pine_api_key = os.environ.get("PINECONE_API_KEY")
            pine_env = os.environ.get("PINECONE_ENV")
            pinecone.init(api_key=pine_api_key, environment=pine_env)

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    metric="cosine",
                    dimension=1536,  # 1536 dim of text-embedding-ada-002
                )
            # if memory_index_name not in pinecone.list_indexes():
            #     pinecone.create_index(
            #         name=memory_index_name,
            #         metric="cosine",
            #         dimension=1536,  # 1536 dim of text-embedding-ada-002
            #     )
            # wait for index to be initialized
            while not pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)

            docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        # query = "How many sections are there in the Indian Penal Code?"
        # docs = docsearch.similarity_search(query)

        # memory_index = pinecone.Index("chat-memory-index")
        # vectorstore = Pinecone(memory_index, embeddings.embed_query, "text")
        # retriever = vectorstore.as_retriever()

        return docsearch

    def doc_handler(self):
        doc_path = str("./penal_code.pdf")
        loader = PyPDFLoader(doc_path)
        documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )
        docs = text_splitter.split_documents(documents)
        return docs

    def prompts_templates(self, tools):
        # Set up the base template
        template = """You are a helpful conversational assistant on Indian Law and Governemnt policy related topics. Answer the following questions as best you can, you have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        This is the history of our conversation:
        {history}

        Begin!

        Question: {input}
        {agent_scratchpad}"""

        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "history"],
        )

        output_parser = CustomOutputParser()
        return prompt, output_parser

    def main(self, locol_vectorstore: bool, openai_api_key: str) -> AgentExecutor:
        docs = self.doc_handler()

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

        index_name = "indian-penal-code"  # change to pick up from file name
        docsearch = self.vector_store(docs, index_name, embeddings, locol_vectorstore)
        penal_code = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
        )
        print("vectore store retriver tool setup")

        # conversational chain
        memory = ConversationBufferWindowMemory(k=6)
        # conversation_chain = ConversationChain(llm=llm, memory=memory)

        # setup for csv agent
        # setup for sql agent
        tools = self.tools_bag(penal_code)
        prompt, output_parser = self.prompts_templates(tools)
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        print("main llm chain setup")

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
        )
        return agent_executor

    # agent_chain = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #     verbose=True,
    #     memory=memory,
    # )


if __name__ == "__main__":
    load_dotenv()

    agent_executor = Conversation.main(True, os.environ.get("OPENAI_API_KEY"))

    agent_executor.run(
        input="hi, can you tell me how many sections and chapters are there in the Indian Penal code ?"
    )
    agent_executor.run(input="what are those 4 penal codes you found?")
    agent_executor.run(
        input="elaborate the 6th penal code with an example- 6. Definitions in the Code to be understood subject to exceptions. "
    )
    agent_executor.run("Search the web for what is the date today ?")
    agent_executor.run(input="Hi, my name is Azhar Majeed")

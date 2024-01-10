import os
from langchain.vectorstores import Chroma
from langchain import OpenAI, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=500)

persist_directory = 'db_langchain'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()
retriever.search_kwargs = {"n": 5, "k": 10}
print(vectordb.similarity_search("How do I use the Agent class in Langchain?"))

class VectorDBQueryAndFilterTool(BaseTool):
    def __init__(self, vectordb, filter_chain):
        self.vectordb = vectordb
        self.filter_chain = filter_chain

    def run(self, query):
        # Query the VectorDB
        documents = self.vectordb.query(query)

        # Filter the documents according to your criteria
        filtered_documents = self.filter_chain(documents)

        # Return the filtered documents
        return filtered_documents
# summary_chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce")
# summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)


qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="map_reduce", retriever=retriever)

# print(qa.run("Explain how Agents work in Langchain."))

# memory = ConversationBufferMemory()
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=memory)


# def qa_chain_wrapper(question: str):
#     return qa({"question": question, "chat_history": []})


# tools = [
#     Tool(
#         name="LangChain codebase",
#         func=qa.run,
#         description="Useful to answer questions with information from the python Langchain documentation.",
#     )
# ]

# qa.run(
#     """
#     Write a python tutorial on how the Agent class works in Langchain. Your tutorial should
#     be aimed at information retrieval from a Chroma Database and the best way to 
#     generate a body of text systhesizing data from multiple db queries.
#     """
# )

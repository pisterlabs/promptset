# Import things that are needed generically
from langchain.tools import BaseTool, Tool
from typing import Any, Optional
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from llms.azure_llms import create_llm
from uploaders.main import get_pdf_text, get_text_chunks, get_vectorstore
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

tool_llm = create_llm()

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

qa: Any
vectorstore: None

def create_qa_retriever(docs, type="azure", database="FAISS"):
    global qa
    global vectorstore
    # Recieve PDF text
    raw_text = get_pdf_text(docs)
    # Split data into text chunks
    text_chunks = get_text_chunks(raw_text)
    # Create vector store
    vectorstore = get_vectorstore(text_chunks, type=type, database=database)
    #qa = RetrievalQA.from_chain_type(llm=tool_llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory)

class qa_retrieval_tool(BaseTool):
    name = "Local Documents"
    description = "use for getting contextually relevant and accurate information to then use for creating a detailed answer"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            global vectorstore
            docs = vectorstore.similarity_search(query)
            chain = load_qa_chain(tool_llm, chain_type="stuff")
            #vectorstore.similarity_search(query)[0].page_content
            return chain.run(input_documents=docs, question=query)
        except:
            return "Tool not available for use."


    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
qa_retrieve = qa_retrieval_tool()
qa_retrieval_tool = Tool(
    name = "Local Documents",
    description = "use for getting contextually relevant and accurate information to then use for creating a detailed answer",
    func= qa_retrieve.run
    )
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from utils import read_funcs
import json
import tools

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class Retriever:
    def __init__(self, debug=False, max_tools=10):
        self.All_tools = []
        self.max_tools = max_tools
        self.llm = ChatOpenAI(temperature=0)
        self.debug = debug
        search_apis = read_funcs("tools.py")
        for name in search_apis:
            self.All_tools.append(getattr(tools, name))

        docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(self.All_tools)]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        self.retriever = vector_store.as_retriever(k=self.max_tools)

    def get_tools(self, query):
        messages = []
    
        with open("prompts/prompting.json", "r") as f:
            data = json.load(f)['paraphrase']
            messages.append(SystemMessage(content=data["system"]))
            messages.append(AIMessage(content=data["input"]))
            messages.append(HumanMessage(content=data["output"]))

        messages.append(HumanMessage(content=query))
        paraphrase = self.llm(messages).content
        docs = self.retriever.get_relevant_documents(query + "\n" + paraphrase)
        tools = [self.All_tools[d.metadata["index"]] for d in docs]
        
        return tools

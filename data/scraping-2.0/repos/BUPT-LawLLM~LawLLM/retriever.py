import os
import yaml
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


class Retriever:
    def __init__(self, relevant_k=8, source=None):
        self.relevant_k = relevant_k
        self.source = source

        with open("config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        self.embedding_model_name = config["EMBEDDING_MODEL_NAME"]
        self.vector_store_path = config["VECTORSTORE_PATH"]
        self.gpt_model = config["MODEL_NAME"]

        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
        os.environ["SEARCHAPI_API_KEY"] = config["SEARCHAPI_API_KEY"]

    def query_vector_store(self, query):
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        vector_store = FAISS.load_local(folder_path=self.vector_store_path, embeddings=embeddings)
        search_result = vector_store.similarity_search_with_score(query=query, k=self.relevant_k)
        
        retrieved_laws = [] 
        for (doc, l2) in search_result:
            retrieved_laws.append(doc.page_content)
            print(doc.page_content.replace("\n", ""), ":", l2)

        return "\n".join(retrieved_laws)
    
    def query_searchapi(self, query, verbose=True):
        llm = ChatOpenAI(temperature=0, model_name=self.gpt_model)
        tools = load_tools(["searchapi"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
        return agent.run(query)

    # def query_all(self, query):
    #     retrieved_laws_local = query_vector_store(query)
    #     retrieved_laws_web = query_searchapi(query)
    #     return {"local": retrieved_laws_local, "web": retrieved_laws_web}
    
    def query(self, query, verbose=True):
        if self.source == "web":
            return self.query_searchapi(query, verbose=verbose)
        elif self.source == "local":
            return self.query_vector_store(query)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    law_retriever = Retriever(relevant_k=5)
    # retrieved_laws = law_retriever.query_searchapi(query="《中华人民共和国民事诉讼法》第一百四十四条 的内容是什么？")
    retrieved_laws = law_retriever.query_vector_store(query="《中华人民共和国民事诉讼法》第一百四十四条 的内容是什么？")
    print(retrieved_laws)
    
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
import pinecone
import os

os.environ['OPENAI_API_KEY']="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"
os.environ['PINECONE_ENV']="us-west4-gcp-free"
os.environ['PINECONE_API_KEY']="6b699f4f-5b1d-471a-adbe-918747981c1b"


class Pinecone_client:
    def __init__(self,embeddings):
        PINECONE_API_KEY=os.environ['PINECONE_API_KEY']
        PINECONE_ENV=os.environ['PINECONE_ENV']
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index_name = "neet-bot"
        
        
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)  # Adjust the dimension as per your document representation
        self.index = pinecone.Index(index_name)
        self.embeddings=embeddings
        
    def get_retriever(self):
        docsearch=Pinecone(self.index, self.embeddings.embed_query, "text")
        return docsearch


class LLM:
    def __init__(self):
        self.embeddings=HuggingFaceEmbeddings()
        api_key=os.environ['OPENAI_API_KEY']
        self.model=ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo',api_key=api_key)
        
    def set_prompt(self,prompt=None):
        if not prompt:
            prompt="You are a bot that answers questions related to NEET (National Eligibility Cum Entrence Test) Biology syllabus based on context provided below." 
        prompt_template= prompt + """
        {context}
        question : {question}
        """

        self.PROMPT=PromptTemplate(
            template=prompt_template,
            input_variables=['context','question']
        )
        
        
class RagBot:
    def __init__(self):
        self.llm=LLM()
        self.pinecone_client=Pinecone_client(self.llm.embeddings)
        self.retriever=self.pinecone_client.get_retriever()
        self.prompt=self.llm.set_prompt()
        
        
        
        
        self.bot=RetrievalQA.from_chain_type(
            llm=self.llm.model,
            chain_type='stuff',
            retriever=self.retriever.as_retriever(search_type='mmr'),
            chain_type_kwargs={'prompt':self.prompt,},
            return_source_documents=True
        )
        
        
        
        
        

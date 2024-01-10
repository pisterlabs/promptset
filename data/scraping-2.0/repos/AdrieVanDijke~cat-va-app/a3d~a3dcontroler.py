import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai
from openai import OpenAI
import streamlit as st
from a3d.a3d_teksten import A3DTeksten

# Maak een client-object
client = OpenAI(api_key=openai.api_key)

class A3DControler:
    def __init__( self, a3dmod ):
        self.a3dmod = a3dmod
        self.a3dtekst = A3DTeksten()

    # MAIN ========================================================================
    # vraag het de Pinecone database =========================
    def ask_the_database(self, query):   
        vectorstore = self.get_database()   
        prompt_template = self.a3dtekst.get_db_prompt_template() 
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        llm = self.get_llm()       
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type='stuff', 
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs=chain_type_kwargs
        )
        result = qa.run(query)               
        return result

    # vraag het de OpenAI API -> fine-tuned model ============
    def ask_model(self, input_text):
        llm_response = self.get_tuned_model(input_text)
        response = llm_response.choices[0].message.content if hasattr(llm_response.choices[0].message, 'content') else ""
        return response
           
    # WORKERS =====================================================================
    # maak en onderhoud verbinding met de Pinecone database ==
    @st.cache_resource
    def get_database(_self):
        pinecone.init(api_key=_self.a3dmod.pinecone_api_key, environment=_self.a3dmod.pinecone_environment)
        index = pinecone.Index(_self.a3dmod.pinecone_index_name)
        embeddings = OpenAIEmbeddings() 
        vectorstore = Pinecone(index, embeddings, "text")  
        return vectorstore
    
    # maak en onderhoud verbinding met de OpenAI API ==========
    @st.cache_resource
    def get_llm(_self):
        llm = ChatOpenAI(model_name=_self.a3dmod.aimodel, temperature=_self.a3dmod.temperature, max_tokens=_self.a3dmod.max_tokens)
        return llm

    # Haal een getraind model op bij OpenAI ===================
    def get_tuned_model(self, input_text):
        system_prompt = self.a3dtekst.get_qa_system_prompt()
        completion = client.chat.completions.create(       
            model=self.a3dmod.finemodel,
            temperature=self.a3dmod.fine_temperature,
            max_tokens=self.a3dmod.fine_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
        )
        return completion

        
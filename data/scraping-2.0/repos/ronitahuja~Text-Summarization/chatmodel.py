import os
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator=' ',
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    text_chunks=text_splitter.split_text(text)
    return text_chunks

def vector_storage(text_chunks):
     model_name = "hkunlp/instructor-large"
     model_kwargs = {'device': 'cpu'}
     encode_kwargs = {'normalize_embeddings': True}
     embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
     vector_store=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
     return vector_store
 
def get_conversation_chain(vector_store):
     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
     memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
     conversational_chain=ConversationalRetrievalChain.from_llm(llm=llm,memory=memory,retriever=vector_store.as_retriever())
     return conversational_chain
     
def fun(text):
    text_chunks= get_text_chunks(text)
    vector_store=vector_storage(text_chunks)
    conversation=get_conversation_chain(vector_store)
    return conversation

if __name__ == '__main__':
    fun()


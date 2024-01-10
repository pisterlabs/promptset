from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import StuffDocumentsChain
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import load_chain

########################
from prompts import system_message_prompt, human_message_prompt
########################
# ChatGPT API 
os.environ['OPENAI_API_KEY'] = "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"

DATASET_PATH = "../datasets/Data/Transcripts/"
VECTORSTORE_FILE_NAME = "vectorstore.faiss"
MODEL_NAME = "gpt-3.5-turbo-16k"
# Load Documents
def load_documents():
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(DATASET_PATH,show_progress=True, use_multithreading=True, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    print(f"successfully loaded {len(docs)} docs.")
    return docs
    ################
# Split Documents
def split_documents(docs):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"Successfuly split documents into : {len(split_documents)} chunks.")
    return split_documents
################
# Faiss vector store
def create_vector_store(split_documents):
    db = FAISS.from_documents(split_documents, OpenAIEmbeddings())
    db.save_local(VECTORSTORE_FILE_NAME)
    return db

    # query = "Hey doctor, I have a headache."
    # docs = db.similarity_search(query)
    # print(f"Found : {len(docs)} chunks for the query : {query}")
    # print(docs[0].page_content)
    # print("=====================================")
    # print(docs[1].page_content)
    # print("=====================================")
    # print(docs[2].page_content)
    # print("=====================================")


def create_chain(db):
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    combine_docs_chain_kwargs = {
          "prompt": ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ]),
    }
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), combine_docs_chain_kwargs=combine_docs_chain_kwargs)
    return qa

def prep_chain():
    # If the vectorstore exists, just use it
    if os.path.exists(VECTORSTORE_FILE_NAME):
        print("Using loaded vectordb")
        db = FAISS.load_local(VECTORSTORE_FILE_NAME, OpenAIEmbeddings())
    else:
        docs = load_documents()
        split_docs = split_documents(docs)
        db = create_vector_store(split_docs)

    chain = create_chain(db)
    return chain
def get_final_analysis(overall_analysis: dict):

    final_analysis_model = ChatOpenAI(model_name="gpt-4", temperature=0)
    messages = [SystemMessage(content="You are a summariser model that can summarise medical diagnoses."), HumanMessage(content=f"Summarise the given diagnostic data into a final report.\n{str(overall_analysis)}")]
    return final_analysis_model(messages).content


def main():
    chain = prep_chain()
    query = """Hey, how is your day going 
    I have a headache.
    How long have you had it?
    For about 2 days now."""
    patient_info = """
    Age: 25,
    Gender: Male,
    Occupation: Student,
    Family medical history: Unknown,
"""
    result = chain({"question": query, "chat_history": [], "patient_info": patient_info})
    print(result)


if __name__ == "__main__":
    main()
from pathlib import Path
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from core.utils import create_prompt_template
# import tiktoken


def get_llm(model_name:str, model_id:str, predictor_params:dict):

    # if model_name == "flan_t5_xl":
    #     return HuggingFaceHub(
    #         repo_id=model_id,
    #         model_kwargs={"temperature":1e-10}
    #     )
    if model_name == "gpt3.5_turbo" or model_name == "text_davinci_003":

        model = OpenAI(model_name=model_id, model_kwargs=predictor_params)
        # encoding = tiktoken.encoding_for_model(model_id)

    return model

    

    raise Exception(f"Invalid model selected.")



def get_retrieval_chain(llm, vectorstore_folder:Path):

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=str(vectorstore_folder), embedding_function=embedding)
    
    chain_type_kwargs = {"prompt": create_prompt_template()}
    llm_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectordb.as_retriever(), 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return llm_retrieval_chain


def get_chain(llm):


    llm_chain = LLMChain(
        prompt=create_prompt_template(),
        llm=llm
    )

    return llm_chain


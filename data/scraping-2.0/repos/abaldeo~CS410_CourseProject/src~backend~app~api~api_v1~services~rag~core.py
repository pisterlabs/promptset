
from app.api.api_v1.services.embedding.core import bulk_load_n_split_docs, create_vectorstore, get_embedding_model
import g4f
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM

from langchain.prompts.prompt import PromptTemplate
import os
from app.core.config import settings, get_settings 
from loguru import logger


def get_bm25_retriever(documents_path="s3://coursebuddy/cs410/transcripts/"):
    from langchain.retrievers import BM25Retriever
    documents = bulk_load_n_split_docs(directory_path=documents_path)
    
    bm25_params = dict(k1= float(settings.BM25_K1), b=float(settings.BM25_B), epsilon=float(settings.BM25_EPSILON))
    bm25_retriever = BM25Retriever.from_documents(documents, bm25_params=bm25_params)
    bm25_retriever.k = int(settings.TOP_K)
    return bm25_retriever

def get_hybrid_retriever(dense_retriever, bm25_weight=None):
    from langchain.retrievers import EnsembleRetriever
    # Create a retriever object
    if not bm25_weight: bm25_weight= float(settings.BM25_WEIGHT)
    dense_weight = 1.0 - bm25_weight
    bm25_retriever = get_bm25_retriever()
    #dense_retriever = get_vectorstore_as_retriever(vectorstore)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], weights=[bm25_weight, dense_weight]
    )
    return ensemble_retriever

def get_vectorstore_as_retriever(vectorstore, search_type=None, search_kwargs=None):
    if not search_type: search_type = settings.SEARCH_TYPE 
    if not search_kwargs: search_kwargs = search_kwargs={'k': int(settings.TOP_K), 'fetch_k': int(settings.FETCH_K),
                                                         'lambda_mult': float(settings.MMR_SIMILARITY)}
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
                   
                   
def get_llm(model=settings.GPT_MODEL_NAME, provider=settings.LLM_PROVIDER, **kwargs):
    provider = str(provider).lower() 
    model = str(model).lower()
    if "g4f" in provider:
        llm: LLM = G4FLLM(
            model=models.gpt_35_turbo, 
            provider=None,
            create_kwargs=kwargs
        )
    elif "azure" in provider:
        from langchain.llms import AzureOpenAI
        kwargs.setdefault("azure_open_api_key", settings.OPENAI_API_KEY)
        kwargs.setdefault("azure_openai_api_endpoint", settings.OPENAI_API_BASE)
        kwargs.setdefault("openai_api_version", settings.OPENAI_API_VERSION)
        kwargs.setdefault("model_version", settings.AZURE_OPENAI_MODEL_VERSION)
        llm: LLM = AzureOpenAI(model_name=model, **kwargs)
    else:
        from langchain.chat_models import ChatOpenAI
        kwargs.setdefault("api_key", settings.OPENAI_API_KEY)
        llm: LLM = ChatOpenAI(model_name=model, **kwargs)
    return llm 

def get_streaming_llm(model=settings.GPT_MODEL_NAME, provider=settings.LLM_PROVIDER, callbacks=None, **kwargs):
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    if callbacks is None: callbacks = [StreamingStdOutCallbackHandler()]
    return get_llm(model=model, provider=provider, callbacks=callbacks, stream=True, **kwargs)  

def get_qa_chain_prompt_template(context_var_name='context', question_var_name='question'):
    import textwrap     
    QA_CHAIN_PROMPT=textwrap.dedent("""You are a helpful assistant, you will use the provided context to answer user questions. 
    Read the given context before answering questions and think step by step.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {""" + context_var_name + """}
    Question: {""" + question_var_name + """}
    Helpful Answer:""")
    logger.debug(QA_CHAIN_PROMPT)
    prompt_template = PromptTemplate(template=QA_CHAIN_PROMPT, input_variables=[context_var_name, question_var_name])
    return prompt_template   

def get_qa_chain_with_sources_prompt_template(context_var_name='summaries', question_var_name='question'):
    import textwrap 
    QA_CHAIN_PROMPT=textwrap.dedent("""You are a helpful assistant, you will use the provided context to answer user questions. 
    Read the given context before answering questions and think step by step.
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    CONTEXT: 
    =========
    {""" + context_var_name + """}
    =========
    QUESTION: {""" + question_var_name + """}
    ANSWER:""")
    logger.debug(QA_CHAIN_PROMPT)
    prompt_template = PromptTemplate(template=QA_CHAIN_PROMPT, input_variables=[context_var_name, question_var_name])
    return prompt_template 

def create_qa_chain(llm, vectorstore):
    from langchain.chains import RetrievalQA

    prompt_template = get_qa_chain_prompt_template()
    vector_db_retriever= get_vectorstore_as_retriever(vectorstore)
    retriever = vector_db_retriever if not settings.ENABLE_HYBRID_SEARCH else get_hybrid_retriever(vector_db_retriever)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retriever ,
                                           return_source_documents=True, 
                                           input_key="question", 
                                           output_key="answer",
                                           chain_type_kwargs={"prompt": prompt_template})
    return qa_chain 

def create_qa_chain_with_sources(llm, vectorstore):
    from langchain.chains import RetrievalQAWithSourcesChain

    prompt_template = get_qa_chain_with_sources_prompt_template()
    vector_db_retriever= get_vectorstore_as_retriever(vectorstore)
    retriever = vector_db_retriever if not settings.ENABLE_HYBRID_SEARCH else get_hybrid_retriever(vector_db_retriever)
    logger.debug("HYBRID SEARCH ENABLED: {}".format(settings.ENABLE_HYBRID_SEARCH))

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, 
                                                           retriever=retriever,
                                                           #reduce_k_below_max_tokens=False,  
                                                           #max_tokens_limit=3375,  
                                                           return_source_documents=True,
                                                           chain_type_kwargs={"prompt": prompt_template}
                                                           )
    return qa_chain 

def create_conversation_chain(llm, vectorstore, memory, callbacks=None):
    from langchain.chains import ConversationalRetrievalChain  
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chains.llm import LLMChain
    from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
                                                        #,QA_PROMPT

    prompt_template = get_qa_chain_with_sources_prompt_template()
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=prompt_template)
    vector_db_retriever= get_vectorstore_as_retriever(vectorstore)
    retriever = vector_db_retriever if not settings.ENABLE_HYBRID_SEARCH else get_hybrid_retriever(vector_db_retriever)
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,        
        combine_docs_chain=doc_chain,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        callbacks=callbacks,
        rephrase_question=False, 
        verbose=True, #langchain.globals.get_verbose().
    )
    return qa 

def create_memory(chat_history=None):
    from langchain.memory import ConversationBufferMemory, ChatMessageHistory
    if chat_history is None: chat_history = ChatMessageHistory()
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True, 	  
                                    output_key='answer', chat_memory=chat_history)

def create_redis_history(session_id, url=settings.EMBEDDING_REDIS_URL, token=settings.EMBEDDING_REDIS_TOKEN, ttl=10):
    from langchain.memory.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
    history = UpstashRedisChatMessageHistory(url=url, 
                                             token=token, 
                                             ttl=ttl, 
                                             session_id=session_id)
        
    # from langchain.memory import RedisChatMessageHistory
    # history - RedisChatMessageHistory(session_id)
    return history 

def create_postgres_history(session_id, connection_string=settings.SQLALCHEMY_DATABASE_URI, table_name="chat_history"):
    from langchain.memory.chat_message_histories.postgres import PostgresChatMessageHistory
    return PostgresChatMessageHistory(connection_string=connection_string, session_id=session_id, table_name=table_name)

def format_qa_response(chain_output):
    logger.debug(chain_output)
    answer = chain_output.get("answer")
    citations = format_citations(chain_output)            
    return f"{answer} \n SOURCES: \n {citations}"

def format_citations(chain_output):
    sources = chain_output.get('sources')
    if sources:
        citations = []        
        source_documents = chain_output.get('source_documents', [])
        for document in source_documents:
            source = document.metadata.get('source')
            if source in sources:
                lecture_title = document.metadata.get('lecture_title')
                lecture_number = document.metadata.get('lecture_number')
                doc_name =  os.path.basename(source)
                document_citation = f"{lecture_number} - {lecture_title} ({doc_name})"
                citations.append(document_citation)
        return "\n".join(set(citations))
    return sources


def main():
    import langchain
    langchain.globals.set_verbose(True)
    
    model = get_embedding_model()
    
    vector_db = create_vectorstore(embedding_model=model, recreate=False)
  
    g4f.debug.logging = False # enable logging
    g4f.check_version = False # Disable automatic version checking
    print(g4f.version) # check version
    print(g4f.Provider.Ails.params)  # supported args
    llm = get_llm()
    history = create_redis_history(session_id="test")
    memory = create_memory(history)
    qa = create_conversation_chain(llm, vector_db, memory)
    # qa = create_qa_chain_with_sources(llm, vector_db)
    answer=qa({"question":"What is text clustering"})
    print(answer)
    answer=qa({"question":"What does similiarty mean in this context?"})
    print(answer)
    # llm: LLM = get_streaming_llm()
    # response = llm("Write me a song about sparkling water.")
    # for chunk in response: 
    #     print(chunk, end="", flush=True)
    # qa = create_qa_chain_with_sources(llm, vector_db)
    # answer=qa({"question":"What is text clustering"})
    # print(format_qa_response(answer))

if __name__ == "__main__":
    main() 
#%%
from rag_embedding import get_retriever_embeddings, get_generator_embeddings
from rag_prompting import set_custom_prompt
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35, load_llm_ctra_llama2_13b, load_llm_tokenizer_llama2_13b_hf
from rag_chains import retrieval_qa_chain_from_local_db, final_result
from rag_vectorstore import load_chroma_db, load_local_faiss_vector_database


from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers # to use CPU only
from langchain.llms import HuggingFacePipeline


QUERY = "What is the probability of you being so much taller than the average? "

DB_FAISS_PATH = "vectorstores/db_faiss"
DB_CHROMA_PATH = "vectorstores/db_chroma"


#%% #######################################################################
def test_get_generator_embeddings():
    embed_model = get_generator_embeddings()
    assert isinstance(embed_model.model_name, str) , "Failed getting the embedding model, check get_generator_embeddings()"
    print(f'Embedding model name: {embed_model.model_name}')
    return embed_model

test_embed_model = test_get_generator_embeddings()

#%% VECTOR STORE ############################################################

def test_read_db_faiss():
    db = load_local_faiss_vector_database(DB_FAISS_PATH, test_embed_model)
    assert isinstance(db, FAISS) , "Failed getting the db, check test_read_db()"
    return db

def test_read_db_chroma():
    db = load_chroma_db(test_embed_model)
    assert isinstance(db, Chroma) , "Failed getting the db, check test_read_db()"
    return db

test_db = test_read_db_faiss()
test_db = test_read_db_chroma()

#%%
def test_set_custom_prompt():
    prompt = set_custom_prompt()
    assert isinstance(prompt, PromptTemplate) , "Failed getting the prompt, check test_custom_prompt()"
    return prompt

test_prompt = test_set_custom_prompt()
print(
    test_set_custom_prompt().format(
        context="The sky is blue.",
        query="What is the colour of the sky?"
    )
)


#%% LLMS TESTING #########################################################
def test_load_llm_ctra_llama27b():
    llm = load_llm_ctra_llama27b()
    assert isinstance(llm, CTransformers) , "Failed getting the llm, check test_load_llm_ctra_llama27b()"
    return llm

#%%
def test_load_llm_ctra_llama2_13b():
    llm = load_llm_ctra_llama2_13b()
    assert isinstance(llm, CTransformers) , "Failed getting the llm, check test_load_llm_ctra_llama27b()"
    return llm

test_llm = test_load_llm_ctra_llama2_13b()

#%%
def test_load_llm_tokenizer_llama2_13b_hf():
    llm = load_llm_tokenizer_llama2_13b_hf()
    assert isinstance(llm, HuggingFacePipeline) , "Failed getting the llm, check test_load_llm_ctra_llama27b()"
    assert (llm.model_id == 'gpt2')
    return llm

test_llm = test_load_llm_tokenizer_llama2_13b_hf()

#%% 

def test_load_llm_gpt35():
    llm = load_llm_gpt35()
    assert isinstance(llm, ChatOpenAI) , "Failed getting the llm, check test_load_llm_gpt35()"
    return llm

test_llm = test_load_llm_gpt35()
#%%
def test_retrieval_qa_chain_from_local_db():
    llm = test_llm
    prompt = set_custom_prompt()
    db = FAISS.load_local("vectorstores/db_faiss", get_generator_embeddings())

    qa_chain = retrieval_qa_chain_from_local_db(llm, prompt, db)
    assert isinstance(qa_chain, RetrievalQA) , "Failed getting the qa_chain, check test_retrieval_qa_chain_from_local_db()"
    return qa_chain


test_qa_chain = test_retrieval_qa_chain_from_local_db()
response = test_qa_chain({"query": QUERY})

# %%

def test_final_result():
    response = final_result(test_qa_chain, QUERY)
    return response

test_chain_response = test_final_result()
print(test_chain_response)



# imports
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA


# cosntants
DIR_DATA = "/home/javaprog/Data/"
DIR_DOCS = DIR_DATA + "ML/Llama2Test/PPARG/Docs"
DIR_VECTOR_STORE = DIR_DATA + "ML/Llama2Test/PPARG/VectorStore"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-7b-chat.ggmlv3.q8_0.bin"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-13b-chat.ggmlv3.q8_0.bin"
PROMPT = """Use the following piece of information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


# methods
def set_prompt(prompt, log=False):
    '''
    returns the prompt to use
    '''
    result_prompt = PromptTemplate(template=prompt, input_variables=['context', 'question'])

    return result_prompt

def load_llm(file_model, log=False):
    if log:
        print("loading model: {}".format(file_model))

    llm = CTransformers(
        model=file_model,
        model_type = "llama",
        max_new_tokens = 512,
        # temperature = 0.1
        temperature = 0.0
    )

    if log:
        print("loaded model from: {}".format(file_model))

    return llm

def get_qa_chain(llm, prompt, db, log=False):
    '''
    get the langchain
    '''
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def get_qa_bot(dir_db, file_model, prompt, log=False):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(dir_db, embeddings)

    llm = load_llm(file_model=file_model, log=log)

    prompt_qa = set_prompt(prompt=prompt, log=log)

    chain_qa = get_qa_chain(llm=llm, prompt=prompt_qa, db=db, log=log)

    return chain_qa

def get_inference(question, chain_qa, log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using query: {}".format(question))
    result = chain_qa({'query': question})

    # if log:
    #     print("got result: {}".format(result))

    # return
    return result


# main
if __name__ == "__main__":
    # initialize
    dir_db = DIR_VECTOR_STORE
    file_model = FILE_MODEL
    prompt = PROMPT

    # get the langchain
    print("creating langchain")
    chain_qa = get_qa_bot(dir_db=dir_db, file_model=file_model, prompt=prompt, log=True)

    # do inference
    question = "what are the top 10 diseases associated with the gene PPARG?"
    response = get_inference(question=question, chain_qa=chain_qa, log=True)
    print("got response: \n\n{}".format(response))

    # do inference
    question = "how does the gene PPARG contribute to type 2 diabetes?"
    response = get_inference(question=question, chain_qa=chain_qa, log=True)
    print("got response: \n\n{}".format(response))

    # # do inference
    # question = "what are an KPs"
    # response = get_inference(question=question, chain_qa=chain_qa, log=True)
    # print("got response: \n\n{}".format(response))



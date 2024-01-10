import os
from dotenv import load_dotenv
load_dotenv('../.env')

from pprint import pprint
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import HuggingFaceHub, PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import DeepLake

VERBOSE = False
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def get_results(query, db):

    retriever = db.as_retriever()
    
    docs = db.similarity_search_with_score(query, k=10)

    print('Results')
    pprint(docs)
    print()


    template = """Question: {question}
        Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # --------
    openai_llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_TOKEN'])

    chain = RetrievalQAWithSourcesChain.from_chain_type(openai_llm, chain_type="stuff", retriever=retriever, verbose=VERBOSE)
    print(f'OpenAI gpt-3.5-turbo LLM')
    docs_hf = chain({"question": query}, return_only_outputs=True)
    pprint(docs_hf)
    print()


    # # --------
    # hf_llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-xl", 
    #     model_kwargs={"temperature":0.1, "max_length":200},
    #     huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    # )

    # chain = RetrievalQAWithSourcesChain.from_chain_type(hf_llm, chain_type="stuff", retriever=retriever, verbose=VERBOSE)


    # print(f'HF LLM: google/flan-t5-xl')
    # docs_hf = chain({"question": query}, return_only_outputs=True)
    # pprint(docs_hf)
    # print()

    

if __name__ == '__main__':

    embeddings = HuggingFaceEmbeddings()

    # deeplake_path = 'hub://anudit/test_csv'
    # db = DeepLake(dataset_path=deeplake_path, token=os.environ['DEEPLAKE_API_TOKEN'], embedding_function=embeddings, read_only=True)

    db = FAISS.load_local("./schemes_faiss_index", embeddings)
    
    query = 'programs for upliftment of women'
    while query!='q':
        
        query = input()

        get_results(query, db)


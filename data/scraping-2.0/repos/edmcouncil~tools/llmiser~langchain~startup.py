from time import time

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from llmiser.langchain.config import EMBEDDING_MODEL


def answer_question(question: str, store_path: str) -> tuple:
    vectorstore = FAISS.load_local(folder_path=store_path, embeddings=EMBEDDING_MODEL)
    # llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, max_tokens=4096)
    llm = Bedrock(model_id="anthropic.claude-instant-v1")
    template = """You are now a conversational assistant whose task is to answer questions about financial concepts only on the basis of the FIBO ontology.
    Answer the questions only using the context.
    Context: {context}
    Question: {question}
    Answer:"""
    chain_prompt = PromptTemplate.from_template(template)
    
    retrieval_qa = \
        RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 16}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': chain_prompt})
    start_time = time()
    response = retrieval_qa({"query": question})
    answer = response['result']
    end_time = time()
    elapsed_time = str(end_time - start_time)
    sources = list()
    for source_document in response['source_documents']:
        source = source_document.metadata['source']
        sources.append(source)
    
    return answer, elapsed_time, sources

# create_and_persist_store(input_docs_folder_path='../gpt_files/', store_path='index/')
answer, elapsed_time, sources = answer_question(question='What are types of loans?', store_path='index/')
print(answer)
print(elapsed_time)
print(sources)
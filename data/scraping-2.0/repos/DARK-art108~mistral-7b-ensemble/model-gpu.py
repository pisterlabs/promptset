from langchain.llms import HuggingFacePipeline
import langchain
from ingest import create_vector_db
from langchain.cache import InMemoryCache
from langchain.schema import prompt
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DB_FAISS_PATH = 'vectorstoredb/db_faiss'


langchain.llm_cache = InMemoryCache()

PROMPT_TEMPLATE = '''
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:
'''
handler = StdOutCallbackHandler()
def set_custom_prompt():
    input_variables = ['context', 'question']
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=input_variables)
    return prompt

def load_retriever():
    return create_vector_db

def load_llm():
    model_name = 'TheBloke/Mistral-7B-Instruct-v0.1-GPTQ'
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-8bit-32g-actorder_True")

    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast = True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def retrieval_qa_chain(llm, prompt, retriever):
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    verbose=True,
    callbacks=[handler],
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
    )
    return qa_chain

def final_result(query):
    llm = load_llm()
    retriever = load_retriever()
    prompt = set_custom_prompt()
    qa_result = retrieval_qa_chain(llm, prompt, retriever=retriever)
    response = qa_result({"query": query})
    return response

if __name__ == "__main__":
    query = "What is the description of the accord form?"
    final_result(query=query)



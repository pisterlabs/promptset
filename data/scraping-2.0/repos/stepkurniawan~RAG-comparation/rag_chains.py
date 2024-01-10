from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain



def retrieval_qa_chain_from_local_db(llm, template_prompt, vectorstore):
    # qa_chain_prompt  = PromptTemplate.from_template(template_prompt)

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', # you can also change this to map reduce
        retriever = vectorstore.as_retriever(search_kwargs = {'k':3}),
        return_source_documents = True,        # retriever will ensure that llm will retrieve the information from the document that we have
        # chain_type_kwargs = {"prompt": qa_chain_prompt} 
    )
    return qa_chain


#QA Model Function
def qa_bot( vectorstore, llm, qa_prompt):
    qa_bot = retrieval_qa_chain_from_local_db(llm, qa_prompt, vectorstore)

    return qa_bot

def qa_with_sources(llm, prompt_template, db):  
    chain_type_kwargs = {"prompt": prompt_template}
  
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

def final_result(qa_chain, query):
    response = qa_chain({'query': query})
    return response

#############
# LOAD QA CHAINS

def chain_with_docs(llm, unique_docs, question):
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain({"input_documents": unique_docs, "question": question}
                    #  ,return_only_outputs=True
                     )
    return response

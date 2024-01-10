"""
https://levelup.gitconnected.com/langchain-for-multiple-pdf-files-87c966e0c032
"""

from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from glob import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed 
from pickle import dump, load

from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.prompt import PromptTemplate


def parallelize(data, func, num_of_processes=8):
    return Parallel(n_jobs=num_of_processes)(delayed(func)(i) for i in tqdm(data))

def try_load(this_path):
    try:
        loader = PyPDFLoader(this_path)
        docs = loader.load()
        return docs
    except:
        print(f'Load failed : {this_path}')
        return None


############################################################
## Generate Docs
############################################################
docs_path = '/media/bigdata/projects/istyar/data/abu_zotero'
file_list = glob(os.path.join(docs_path, "*"))
vector_persist_dir = '/media/bigdata/projects/katzGPT/vector_store'
docs_output_dir = '/media/bigdata/projects/katzGPT/docs'
docs_output_path = os.path.join(docs_output_dir, 'docs.pkl')

if not os.path.exists(vector_persist_dir):
    os.makedirs(vector_persist_dir)

if not os.path.exists(docs_output_dir):
    os.makedirs(docs_output_dir)

if not os.path.exists(docs_output_path):
    docs_list = parallelize(file_list, try_load, num_of_processes=24)
    # Drop None
    docs_list = [doc for doc in docs_list if doc is not None]
    # Flatten list
    docs_list = [item for sublist in docs_list for item in sublist]
    # Extract document source from each document
    doc_source = [doc.metadata['source'] for doc in docs_list]
    ## Count length of set of document source
    #len(set(doc_source))
    # Save docs
    with open(docs_output_path, 'wb') as f:
        dump(docs_list, f)
else:
    docs_list = load(open(docs_output_path, 'rb')) 

############################################################
# Generate Embeddings
############################################################

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents([docs_list[0]], embedding=embeddings, 
                              persist_directory=vector_persist_dir)
vectordb.persist()
for doc in tqdm(docs_list):
    vectordb.add_documents([doc])
vectordb.persist()

memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(
        llm = OpenAI(temperature=0.8) , 
        retriever = vectordb.as_retriever(), 
        # return_source_documents=True,
        memory=memory,
        max_tokens_limit = 3500,
        )
query = "Which katz lab papers discuss attractor networks?"
result = pdf_qa({"question": query})
print(result)

##############################

prompt="""
Follow exactly those 3 steps:
1. Read the context below and aggregrate this data
Context : {matching_engine_response}
2. Answer the question using only this context
3. Show the source for your answers
User Question: {question}


If you don't have any context and are unsure of the answer, reply that you don't know about this topic.
"""

question_prompt = PromptTemplate.from_template(
    template=prompt,
    )


question_generator = LLMChain(
        llm=llm,
        prompt=question_prompt,
        verbose=True,
    )

llm = OpenAI(temperature=0)
doc_chain = load_qa_with_sources_chain(llm)

chain = ConversationalRetrievalChain(
    retriever=vectordb.as_retriever(),
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    query_generator=question_generator,
    memory=memory,
    max_tokens_limit = 3500,
)

##############################
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=vector_persist_dir, 
                  embedding_function=embeddings)

template = """You are an AI assistant for answering questions about systems neuroscience, specifically taste processing.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(
        template=template, 
        input_variables=[
                       "question", 
                       "context"
                       ]
        )

llm = OpenAI(temperature=0.8)
question_generator = LLMChain(
        llm=llm,
        prompt=QA_PROMPT,
        verbose=True,
    )

doc_chain = load_qa_with_sources_chain(llm,
                                       chain_type = 'stuff')

memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        output_key="answer",
        return_messages=True)
retriever = vectordb.as_retriever(
        search_kwargs = {"k":5})
pdf_qa = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    max_tokens_limit = 2000,
    rephrase_question = False,
    verbose=True,
)
query = "Where is the gustatory thalamus?" 
result = pdf_qa({"question": query})
for this_key in result.keys():
    if this_key not in ['chat_history', 'source_documents']:
        print()
        print(f"{this_key} : {result[this_key]}")

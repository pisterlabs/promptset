from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.llms import GPT4All
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CONNECTION_STRING, COLLECTION_NAME

local_path = (
    "./models/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"  # replace with your desired local file path
)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = PGVector.from_existing_index(embedding=embeddings, collection_name=COLLECTION_NAME, connection_string=CONNECTION_STRING)
retriever = db.as_retriever(search_kwargs={"k": 1})

llm = GPT4All(model=local_path, max_tokens=2048, n_batch=model_n_batch, backend="gptj", verbose=False)

async def get_answer(query):
    relevant_docs = db.similarity_search(query)
    #qa_chain = RetrievalQA.from_chain_type(relevant_docs, llm=llm, retriever=retriever)
    qa_chain = load_qa_chain(llm, chain_type='stuff')
    res = qa_chain({"input_documents": relevant_docs, "question": query},return_only_outputs=True)
    return res



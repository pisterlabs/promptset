from langchain import hub
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.ollama import Ollama
from langchain.vectorstores import Chroma

persist_directory = '../../local_data/vector_db_persistence'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
llm = Ollama(
    base_url="http://localhost:11434",
    model="llama2",
    verbose=True,
    temperature=0
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
question = """
Any customer having problem/error with enrollment?
Look for "severity": "ERROR".
"""
result = qa_chain({"query": question})
print(result['result'])

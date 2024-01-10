from baby_agi_with_agent import Optional, BabyAGI
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

OBJECTIVE = "What happened to the First Republic Bank? Will the FED take the same action as it did on SVB's failure?"



llm = OpenAI(temperature=0)




# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 1
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)


response = baby_agi({"objective": OBJECTIVE})
print(response)

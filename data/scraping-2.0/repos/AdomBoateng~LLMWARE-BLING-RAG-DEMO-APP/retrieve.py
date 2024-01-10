######IMPORT NEEDED LIBRARIES#######
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

##LOAD EMBEDDINGS TO TRANSLATE THE SENTENCES READ FROM HUMAN READABLE FORMAT TO NUMERICAL CODE FOR THE MACHINE###
embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")

###LOAD VECTOR STORE "INSURANCE_COSINE"
load_vector_store = Chroma(persist_directory = "stores/insurance_cosine", embedding_function = embeddings)

##QUERY THE MODEL##
query = "What is Accelerated death benefits?"

###DISPLAY THE TOP 3 MOST RELEVANT ANSWERS##
docs = load_vector_store.similarity_search_with_score(query = query, k=3)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

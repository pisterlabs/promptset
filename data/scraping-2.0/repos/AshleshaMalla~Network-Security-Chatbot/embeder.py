from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# embedding
import pickle
from langchain.vectorstores import FAISS
# InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

def store_embeddings(docs, embeddings, sotre_name, path):
    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{sotre_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

def load_embeddings(sotre_name, path):
    with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader("./docs", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print("Texts: ", texts[0], len(texts))

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
Embedding_store_path = "./Embedding_store"

store_embeddings(texts,
                 instructor_embeddings,
                 sotre_name='instructEmbeddings',
                 path=Embedding_store_path)

db_instructEmbedd = load_embeddings(sotre_name='instructEmbeddings',
                                    path=Embedding_store_path)

retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("test")
print(docs[0])
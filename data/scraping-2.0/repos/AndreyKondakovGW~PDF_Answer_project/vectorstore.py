from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
OPEN_AI_API_KEY = os.environ['OPENAI_SECRET']

DATABASE_PATH = './data'
def get_vectorstore(database_name):
    path = os.path.join(DATABASE_PATH, database_name)
    if os.path.exists(path):
        embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY)
        #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={'device': 'cpu'})

        return FAISS.load_local(path, embeddings)
    else:
        return None
    
    
def create_vectorstore(chunks, database_name):
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={'device': 'cpu'})
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY)
    db = FAISS.from_texts(chunks, embeddings)
    print('DB created')
    db.save_local(os.path.join(DATABASE_PATH, database_name))
    return db
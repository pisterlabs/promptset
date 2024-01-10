from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings


embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
    )
db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
retriever = db.as_retriever()


def get_relevant_chunk(query) : 
    try :
        return retriever.get_relevant_documents(query)[0].page_content
    except Exception as e : 
        print(e)
        return None

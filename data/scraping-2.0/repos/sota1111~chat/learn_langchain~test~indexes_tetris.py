from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Document Loaderの使い方
loader = PyPDFLoader("https://chat-tetris.s3.ap-northeast-1.amazonaws.com/seigot_tetris.pdf")
pages = loader.load_and_split()
print("pages:",pages[0])

chroma_index = Chroma.from_documents(pages, OpenAIEmbeddings())
docs = chroma_index.similarity_search("「gameover」に関して教えて。", k=3)
for doc in docs:
    print("page:",str(doc.metadata["page"]) + ":", doc.page_content)

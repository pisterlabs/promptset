from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import os
from dotenv import load_dotenv
import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
NAME_SPACE = os.getenv("NAME_SPACE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG")

pdf_data = []
for doc in glob.glob("data/*.PDF"):
    print(doc)
    loader = PyMuPDFLoader(doc)
    loaded_pdf = loader.load()
    for document in loaded_pdf:
        pdf_data.append(document)
#
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pdf_data)
# # texts = [d.page_content for d in documents]
# # for text in texts:
# #     print(text)
# #     break
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)
#
index = pinecone.GRPCIndex(PINECONE_INDEX)
print(index.describe_index_stats())
#
# # embeddings = OpenAIEmbeddings(
# #     disallowed_special=(),
# # )
#
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# docsearch = FAISS.from_documents(docs, embeddings)
docsearch = Pinecone.from_documents(
    documents, embeddings, index_name=PINECONE_INDEX, namespace=NAME_SPACE
)
# # if you already have an index, you can load it like this
# # docsearch = Pinecone.from_existing_index(
# #     PINECONE_INDEX, embeddings, namespace=NAME_SPACE
# # )
#
# query = "履行期限はどこに書いていますか？"
# res = docsearch.similarity_search_with_score(query, k=5)
# for doc, score in res:
#     print(doc.page_content, score)


# chain = RetrievalQAWithSourcesChain.from_chain_type(
#         ChatOpenAI(
#             model_name="gpt-3.5-turbo-16k",
#             temperature=0,
#         ),
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),
#         return_source_documents=True,
# )
# res = chain(query)
# answer = res["answer"]
# source_elements_dict = {}
# source_elements = []
# for idx, source in enumerate(res["source_documents"]):
#     title = source.metadata["title"]
#
#     if title not in source_elements_dict:
#         source_elements_dict[title] = {
#             "page_number": [source.metadata["page"]],
#             "path": source.metadata["file_path"],
#         }
#
#     else:
#         source_elements_dict[title]["page_number"].append(source.metadata["page"])
#
#     # sort the page numbers
#     source_elements_dict[title]["page_number"].sort()
#
# for title, source in source_elements_dict.items():
#     # create a string for the page numbers
#     page_numbers = ", ".join([str(x) for x in source["page_number"]])
#     text_for_source = f"Page Number(s): {page_numbers}\nPath: {source['path']}"
#     print(f"{title}, \n {text_for_source}")

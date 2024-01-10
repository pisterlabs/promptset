from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
directory = './data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print("document :",len(documents))



def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
# print("split :",len(docs))

# print("text :",docs[2].page_content)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
query_result = embeddings.embed_query("Hello world")
print("Len result :",len(query_result))


import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="6efed09f-2e8e-47f6-ac70-6aba2f0b858b",  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)
index_name ="testwebdsi"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


def get_similiar_docs(query,k=1,score=False): #k จำนวนฉบับ
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

# # from langchain.llms import OpenAI
# # model_name= "gpt-4"
# # llm = OpenAI(model_name)

# # from langchain.chains.question_answering import load_qa_chain
# # chain= load_qa_chain(llm, chain_type="stuff")

# def get_answer(query):
#   similar_docs=get_similiar_docs(query)
#   answer = chain.run(input_documents=similar_docs, question=query)
#   return answer

query = "มาตรา๒๑"
similar_docs = get_similiar_docs(query)
print("test :",similar_docs)
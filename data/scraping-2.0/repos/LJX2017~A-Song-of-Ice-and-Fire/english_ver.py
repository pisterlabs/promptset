from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
def load_chinese():
    import json
    with open("a song of ice and fire.json", "r", encoding="utf-8") as file:
        data_dict = json.load(file)

    texts = ''.join(list(data_dict.values()))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.create_documents([texts])

def load_doc(p="chinese_db.txt"):
    with open(p, "r", encoding="utf-8") as file:
        text_data = file.read()
    print(text_data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.create_documents([text_data])
    return text_splitter.split_documents(texts)
docs = load_doc()
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
query_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load it into Chroma
db = Chroma.from_documents(docs, query_embeddings)
print("check point 1")
# # query it
# query = "tell me"
# docs = db.similarity_search(query)
#
# # print results
# print(docs[0].page_content)

def make_qa_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    return RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"fetch_k":10}),
        return_source_documents=True
    )
qa_chain = make_qa_chain()
q = "summerize the plot with given information"
result = qa_chain({"query": q})
print(result["result"],"\nsources:",)
for idx, elt in enumerate(result["source_documents"]):
    print(elt.page_content)
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from git import Repo


repo = Repo.clone_from(
    "https://github.com/hwchase17/langchain", to_path="/tmp/test_repo"
)

loader = GenericLoader.from_filesystem(
    "/tmp/test_repo/langchain",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=200)
texts = python_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(disallowed_special=())
db = Chroma.from_documents(texts, embeddings)


retriever = db.as_retriever(
    search_type="mmr",  # You can also experiment with "similarity"
    search_kwargs={"k": 8},
)


llm = ChatOpenAI(temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

chat_history = []

question = "How can I load a source code as documents, for a QA over code, spliting the code in classes and functions?"
result = qa({"question": question, "chat_history": chat_history})
chat_history.append((question, result["answer"]))
print(result["answer"])
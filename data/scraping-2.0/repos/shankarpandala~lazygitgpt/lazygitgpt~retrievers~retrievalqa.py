from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from lazygitgpt.datasources.repos import read_repository_contents

db = Chroma.from_documents(read_repository_contents(), OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 1000},
)
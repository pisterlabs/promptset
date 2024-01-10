from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.tools.vectorstore.tool import VectorStoreQATool


def get_db_hints_tools(llm: BaseLanguageModel, doc_path: str, doc_store_name: str = "db_hints_store") -> BaseToolkit:
    loader = TextLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=400, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db_doc_store = Chroma.from_documents(
        texts, embeddings, collection_name=doc_store_name, persist_directory=doc_store_name)

    name="database hints"
    description="hints how to query the question from the database. Always use it BEFORE asking a human. Use this tool in english only."
    vector_store_description = VectorStoreQATool.get_description(
        name, description
    )
    
    qa_tool = VectorStoreQATool(
        name=name,
        description=vector_store_description,
        vectorstore=db_doc_store,
        llm=llm,
    )

    return [qa_tool]
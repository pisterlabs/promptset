# model imports
from langchain.document_loaders import JSONLoader  # importar clase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain,
    ConversationChain,
)
from langchain.chains.conversation.memory import ConversationSummaryMemory

# external imports
from rich.console import Console
from dotenv import load_dotenv

# local imports
from utils import get_file_path, get_query_from_user

console = Console()
recreate_chroma_db = False
chat_type = "summary_memory_chat"


# funcion metadata para retornar title, repo_owner, repo_name en la metadata del documento
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["repo_owner"] = record.get("repo_owner")
    metadata["repo_name"] = record.get("repo_name")

    return metadata


def load_documents(path):
    loader = JSONLoader(
        path,
        json_lines=True,  # indica que el archivo es un jsonl
        jq_schema=".",  # indica que el jsonl tiene un solo elemento por linea, más info en https://python.langchain.com/docs/modules/data_connection/document_loaders/json#common-json-structures-with-jq-schema
        content_key="text",  # indica que el contenido del documento está en la llave "text"
        metadata_func=metadata_func,  # indica que la función metadata_func se usará para obtener la metadata del documento
    )
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, length_function=len, chunk_overlap=160
    )

    return text_splitter.split_documents(data)


def get_chroma_db(embeddings, documents, path):
    if recreate_chroma_db:
        console.print("Creating Chroma DB")
        return Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=path
        )
    else:
        console.print("Loading Chroma DB")
        return Chroma(persist_directory=path, embedding_function=embeddings)


def process_qa_query(query: str, llm: ChatOpenAI, retriever: any):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    console.print("[yellow]Thinking...[/yellow]")
    return qa_chain.run(query)


def process_memory_query(
    query: str, llm: ChatOpenAI, retriever: any, chat_history: any
):
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    console.print("[yellow]Thinking...[/yellow]")
    print(f"the chat history is {chat_history}")
    result = conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]


def process_summary_memory_query(query: str, llm: ChatOpenAI, summary_memory: any):
    conversation = ConversationChain(llm=llm, memory=summary_memory, verbose=True)
    console.print("[yellow]Thinking...[/yellow]")
    result = conversation.predict(input=query)
    return result


def run_conversation(vectorstore: VectorStore, chat_type: str, llm: ChatOpenAI):
    console.print(
        "[blue]hi!! what do you want to ask me about transformers and artificial intelligence?[/blue]"
    )

    if chat_type == "qa":
        console.print(
            "[blue]You are using the chat in question-answer mode, so i won't remember the chat history[/blue]"
        )
    elif chat_type == "memory_chat":
        console.print(
            "[blue]You are using the chat in memory mode, so i will remember the chat history[/blue]"
        )
    elif chat_type == "summary_memory_chat":
        console.print(
            "[blue]You are using the chat in summary memory mode, so i will remember a summary of the chat history[/blue]"
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chat_history = []
    summary_memory = ConversationSummaryMemory(llm=llm)

    while True:
        console.print("[blue]You:[/blue]")
        query = get_query_from_user()

        if (query.lower() == "exit") or (query.lower() == "quit"):
            break

        if chat_type == "qa":
            response = process_qa_query(query, llm, retriever)
        elif chat_type == "memory_chat":
            response = process_memory_query(query, llm, retriever, chat_history)
        elif chat_type == "summary_memory_chat":
            response = process_summary_memory_query(
                query, llm, summary_memory=summary_memory
            )

        console.print(f"[red]AI:[/red] {response}")


def main():
    load_dotenv()

    documents = load_documents(get_file_path())
    embeddings = HuggingFaceEmbeddings()

    vectorstore_chroma = get_chroma_db(embeddings, documents, "chroma_docs")
    console.print(f"[green]Documents {len(documents)} loaded[/green]")

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=200,
    )

    run_conversation(vectorstore_chroma, chat_type, llm)


if __name__ == "__main__":
    main()

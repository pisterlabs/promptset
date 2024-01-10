from common import config
from common import embed

# Rich
from rich.console import Console
console = Console()

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def get(override_action: int, documents_path: str, db_path: str, db_name: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    from chromadb.config import Settings

    client_settings = Settings(
        persist_directory=db_path,
        anonymized_telemetry=config.get('DB_TELEMETRY')
    )

    # Check if we can access chroma db
    try:
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        # In continue mode, the selection means to continue with the current database without overriding it.
        if override_action == 1:
            return db
    except AssertionError as error:
        console.print("An error occurred:", type(error).__name__, "–", error, style="red")
        console.print("Please set GPU_ENABLED to False in your .env file, or install CUDA-compatible PyTorch as per the README.", style="red")
        exit(1)

    # Check if already has a collection and update the collection else add a new collection.
    documents = embed.process_documents(documents_path)
    console.print("Cooking up some embeddings 🔥 This might take a hot few minutes... ⏳", style="yellow")

    collection = db.get()
    if not collection['documents']:
        # Create a new database
        console.print(f"Creating a new vectorstore in: {db_path}", style="yellow")

        if documents:
            # Split by chroma maximum
            split_docs_chunked = embed.split_list(documents, 41000)
            for item in split_docs_chunked:
                console.print(f"Adding a new document(s) to database: {db_name}", style="yellow")
                db.add_documents(item)

        Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=db_path,
            client_settings=client_settings
        )

        db.persist()
    else:
        # Update the database
        console.print(f"Appending to the existing vector store in database: {db_path}", style="yellow")

        Chroma.afrom_documents(
            documents,
            embeddings,
            persist_directory=db_path,
            client_settings=client_settings
        )

    return db
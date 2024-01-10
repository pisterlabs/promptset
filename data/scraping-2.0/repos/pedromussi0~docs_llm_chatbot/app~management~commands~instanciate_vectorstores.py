from django.core.management.base import BaseCommand
from app.models import ProcessedDocument
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import os


class Command(BaseCommand):
    help = "Load documents and embeddings from the database and initialize the vector store"

    def handle(self, *args, **options):
        # Retrieve all ProcessedDocument objects from the database
        processed_documents = ProcessedDocument.objects.all()

        # Filter out already processed URLs
        processed_urls = set()  # Track processed URLs
        documents = []
        for doc in processed_documents:
            if doc.url not in processed_urls:
                documents.append(
                    Document(page_content=doc.content, metadata={"source": doc.url})
                )
                processed_urls.add(doc.url)

        # Split the documents into chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Initialize Embeddings and the FAISS vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

        # Prompt the user for the desired process name
        process_identifier = input(
            "name the theme of you processed documents (keep it short): "
        )

        # Create a directory for the current process
        process_directory = os.path.join(
            "app/vectorstores", process_identifier + "embeds"
        )
        os.makedirs(process_directory, exist_ok=True)

        # Save the vector store inside the process directory
        vectorstore.save_local(process_directory, index_name="faiss_index")

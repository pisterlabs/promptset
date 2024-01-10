# Load packages
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader, Docx2txtLoader, JSONLoader, PyPDFLoader, SeleniumURLLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader, WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter, Language
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain import debug as langchain_debug
import os
import openai
import sys


# Declare function
def FindContentInDocuments(question,
                           folder_or_document_filepath,
                           openai_api_key,
                           vectorstore_collection_name="collection",
                           return_vectorstore=False,
                           debug_mode=False,
                           temperature=0.0,
                           chat_model_name="gpt-3.5-turbo",
                           splitter_mode="recursive_text",
                           splitter_chunk_size=1000,
                           splitter_chunk_overlap=100,
                           splitter_separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                           markdown_header_splitters=[
                                ("#", "Header 1"),
                                ("##", "Header 2"),
                                ("###", "Header 3"),
                                ("####", "Header 4"),
                                ("#####", "Header 5"),
                                ("######", "Header 6")
                            ]):
    # Set debug mode
    langchain_debug = debug_mode
        
    # Ensure the splitter mode is valid
    if splitter_mode not in ["character", "recursive_text", "token"]:
        error_message = f"""
        splitter_mode must be one of: character, recursive_text, token, or markdown.
        """
        raise ValueError(error_message)

    # Create list of documents from folder
    if os.path.isdir(folder_or_document_filepath):
        # Get all documents from the folder and its subfolders
        list_of_documents = []
        for root, dirs, files in os.walk(folder_or_document_filepath):
            for file in files:
                # Check if the file is not a folder
                if not os.path.isdir(file):
                    # Create file path
                    file_path = os.path.join(root, file)
                    # Replace backslashes with forward slashes
                    file_path = file_path.replace("\\", "/")
                    # Append to list of documents
                    list_of_documents.append(file_path)
    else:
        list_of_documents = [folder_or_document_filepath]
        
    # Create list to hold all splits
    all_splits = []

    # Iterate through all documents in the folder
    for document_filepath_or_url in list_of_documents:
        # # Set up the text splitter
        if splitter_mode == "character":
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=splitter_chunk_size,
                chunk_overlap=splitter_chunk_overlap,
                length_function=len
            )
        elif splitter_mode == "recursive_text":
            # If the file is a URL, use the HTML splitter
            if document_filepath_or_url.startswith("http"): 
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.HTML, 
                    chunk_size=splitter_chunk_size, 
                    chunk_overlap=splitter_chunk_overlap
                )
            elif document_filepath_or_url.endswith(".md"):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN,
                    chunk_size=splitter_chunk_size, 
                    chunk_overlap=splitter_chunk_overlap
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    separators=splitter_separators,
                    chunk_size=splitter_chunk_size,
                    chunk_overlap=splitter_chunk_overlap,
                    length_function=len
                )
        elif splitter_mode == "token":
            splitter = TokenTextSplitter(
                chunk_size=splitter_chunk_size, 
                chunk_overlap=splitter_chunk_overlap
            )
        elif splitter_mode == "markdown":
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=markdown_header_splitters
            )
        
        # # Set up Loader
        # If the document is a CSV, use the CSVLoader
        if document_filepath_or_url.endswith(".csv"):
            loader = CSVLoader(
                file_path=document_filepath_or_url,
                source_column=csv_source_column
            )
            data = loader.load_and_split()
        # If the document is a Word document, use the WordLoader
        elif document_filepath_or_url.endswith(".docx") or document_filepath_or_url.endswith(".doc"):
            loader = Docx2txtLoader(document_filepath_or_url)
            data = loader.load()
            data = splitter.split_documents(data)
        # If the document is a website, use the UnstructuredURLLoader or SeleniumURLLoader (in case of JavaScript)
        elif document_filepath_or_url.startswith("http"): 
            if url_is_javascript_rendered:
                loader = SeleniumURLLoader(document_filepath_or_url)
                data = loader.load()
                data = splitter.split_documents(data)
            else:
                loader = WebBaseLoader(document_filepath_or_url)
                data = loader.load()
                data = splitter.split_documents(data)
        # If the document is a Markdown file, use the UnstructuredMarkdownLoader
        elif document_filepath_or_url.endswith(".md"):
            loader = UnstructuredMarkdownLoader(document_filepath_or_url)
            data = loader.load()
            data = splitter.split_documents(data)
        # If the document is a JSON, use the JSONLoader
        elif document_filepath_or_url.endswith(".json") or document_filepath_or_url.endswith(".jsonl"):
            # See if JSON is in JSONL format
            if document_filepath_or_url.endswith(".jsonl"):
                json_lines = True
            else:
                json_lines = False
            # The JSONLoader uses a specified jq schema to parse the JSON files. It uses the jq python package. 
            loader = JSONLoader(
                file_path=document_filepath_or_url,
                jq_schema='.content',
                json_lines=json_lines
            )
            data = loader.load_and_split()
        # If the document is a PDF, use the PyPDFLoader
        elif document_filepath_or_url.endswith(".pdf"):
            loader = PyPDFLoader(document_filepath_or_url)
            data = loader.load()
            data = splitter.split_documents(data)
        # If the document is a PowerPoint, use the PowerPointLoader
        elif document_filepath_or_url.endswith(".pptx") or document_filepath_or_url.endswith(".ppt"):
            loader = UnstructuredPowerPointLoader(document_filepath_or_url)
            data = loader.load()
            data = splitter.split_documents(data)
        # If the document is a text file, use the TextFileLoader
        elif document_filepath_or_url.endswith(".txt"):
            loader = TextLoader(document_filepath_or_url)
            data = loader.load()
            data = splitter.split_documents(data)
        # Add to list of splits
        all_splits.extend(data)
        
    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(
        all_splits, 
        embeddings,
        collection_name=vectorstore_collection_name
    )

    # Create description for source
    source_description = "The text the chunk should be one of the following: " + ", ".join(list_of_documents)

    # Set metadata field info 
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description=source_description,
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page number containing the text",
            type="integer",
        ),
    ]

    # Wrap vectorstore in a compressor
    llm = OpenAIChat(
        model_name=chat_model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )
    compressor = LLMChainExtractor.from_llm(llm)

    # Create a compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever()
    )

    # Get relevant documents based on question
    compressed_docs = compression_retriever.get_relevant_documents(question)

    # Print documents in an easily readable format
    for doc in compressed_docs:
        print(f"Document: '{doc.metadata['source']}'")
        print(f"Relevant text: {doc.page_content}")
        try:
            print(f"Page: {doc.metadata['page']}")
            print("\n\n")
        except KeyError:
            print("\n\n")
            continue

    # Reset debug mode
    langchain_debug = False

    # Return the list of documents, and vectorstore if requested
    if return_vectorstore:
        dict_to_return = {
            'VectorStore': vectorstore,
            'RelevantDocuments': compressed_docs
        }
        return(dict_to_return)
    else:
        return(compressed_docs)


# # Test function
# relevant_docs = FindContentInDocuments(
#     question="How do I calculate measure ratings from national benchmarks?",
#     folder_or_document_filepath = "C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/Documentation/NCQA/Methodology",
#     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read(),
#     return_vectorstore=True
# )
# # relevant_docs = FindContentInDocuments(
# #     question="How do I calculate measure ratings from national benchmarks?",
# #     folder_or_document_filepath = "C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/Documentation/NCQA/Methodology/2024-HPR-Methodology_3.30.2023.pdf",
# #     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # )

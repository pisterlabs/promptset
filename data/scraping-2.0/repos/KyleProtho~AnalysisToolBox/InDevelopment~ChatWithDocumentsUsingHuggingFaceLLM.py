from IPython.display import display, HTML, Markdown
import json
import langchain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader, Docx2txtLoader, JSONLoader, PyPDFLoader, SeleniumURLLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader, WebBaseLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter, Language
from langchain.vectorstores import FAISS, DocArrayInMemorySearch, Chroma
import openai
import tiktoken

# Declare the function
def ChatWithDocumentsUsingHuggingFaceLLM(list_of_questions,
                                  huggingface_api_key,
                                  document_filepath_or_url=None,
                                  vectorstore_filepath=None,
                                  query_method="stuff",
                                  temperature=0.0,
                                  max_token_length=4000,
                                  qa_model_repo_id="declare-lab/flan-alpaca-large",
                                  embedding_model_repo_id="sentence-transformers/all-mpnet-base-v2",
                                  verbose=True,
                                  json_field=None,
                                  csv_source_column=None,
                                  url_is_javascript_rendered=False,
                                  vectorstore_collection_name="collection",
                                  return_vectorstore=False,
                                  debug_mode=False,
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
                                  ],
                                  chat_history_name="chat_history"):
    # Set debug mode
    langchain.debug = debug_mode
    
    # Ensure that either a document_filepath_or_url or vectorstore_filepath is provided
    if document_filepath_or_url is None and vectorstore_filepath is None:
        raise ValueError("Either a document_filepath_or_url or vectorstore_filepath must be provided.")
    
    # Ensure that the query method is valid
    if query_method not in ["stuff", "map_reduce", "refine"]:
        error_message = f"""
        query_method must be one of: stuff, map_reduce, or refine.
        - stuff: makes a single query to the model and it has access to all data as context
        - map_reduce: breaks documents into independent chunks and calls, then a final call is made to reduce all responses to a final answer. Best for summarizing really long documents.
        - refine: break documents into chunks, but each response is based on the previous.
        """
        raise ValueError(error_message)
    
    # Ensure the splitter mode is valid
    if splitter_mode not in ["character", "recursive_text", "token"]:
        error_message = f"""
        splitter_mode must be one of: character, recursive_text, token, or markdown.
        """
        raise ValueError(error_message)
    
    # Create embedding function
    embeddings = HuggingFaceHubEmbeddings(
        repo_id=embedding_model_repo_id, 
        task="feature-extraction",
        huggingfacehub_api_token=huggingface_api_key
    )
    
    # If a vectorstore_filepath is provided, load the vectorstore
    if vectorstore_filepath is not None:
        vectordb = Chroma(
            persist_directory=vectorstore_filepath,
            embedding_function=embeddings
        )
    # Otherwise, load the document and create the vectorstore
    elif document_filepath_or_url is not None:
        # Set up the text splitter
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
        
        # Create the vectorstore
        vectordb = Chroma.from_documents(
            data, 
            embeddings,
            collection_name=vectorstore_collection_name
        )
        
        # # Create index of documents
        # if return_vectorstore:
        #     index = VectorstoreIndexCreator(
        #         vectorstore_cls=DocArrayInMemorySearch,
        #         embeddings=embeddings,
        #         collection_name=vectorstore_collection_name
        #     ).from_loaders([loader])

    # Wrap vectorstore in a compressor
    llm = HuggingFaceHub(
        repo_id=qa_model_repo_id, 
        model_kwargs={
            "temperature": temperature, 
            "max_length": max_token_length
        },
        huggingfacehub_api_token=huggingface_api_key
    )
    compressor = LLMChainExtractor.from_llm(llm)

    # Create a compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )
    
    # Iterate through the questions
    if len(list_of_questions)==1:
        # Create the model
        query_retrieval = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type=query_method, 
            retriever=compression_retriever,
            verbose=verbose
        )
        # Ask the question
        response = query_retrieval({'query': list_of_questions[0]})
        response = response['result']
    else:
        # Create memory of conversation
        memory = ConversationBufferMemory(
            memory_key=chat_history_name,
            return_messages=True
        )
        # Create list of responses
        list_of_responses = []
        # Create the model
        query_retrieval = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            verbose=verbose,
            memory=memory
        )
        for i in range(len(list_of_questions)):
            response = query_retrieval({'question': list_of_questions[i]})
            # Add the response to the list
            list_of_responses.append(response['answer'])
        # Join the responses with each question
        response = "\n\n".join([f"**Question:** {list_of_questions[i]}\n\n**Answer:** {list_of_responses[i]}" for i in range(len(list_of_questions))])
    
    # Turn off debugging
    langchain.debug = False  

    # Return the response
    if return_vectorstore:
        dict_to_return = {
            'Response': response,
            'VectorStore': vectordb
        }
        return(dict_to_return)
    else:
        return(response)


# # Test the function
# response = ChatWithDocumentsUsingHuggingFaceLLM(
#     list_of_questions=[
#         """
#         Extract the list of measures.
#         Include the Measure ID, Measure Name, and Weight.
#         Measures are listed under a category (for example "PATIENT EXPERIENCE" or "PREVENTION AND EQUITY"), and a subcategory (for example "Getting Care" or "Children and Adolescent Well-Care").
#         Measure IDs are typically 3-5 characters long, and "PATIENT EXPERIENCE" measures do not have a Measure ID.
#         Ignore the measures listed under "HEDIS/CAHPS/HOS Measures Required for HP Accreditation—Medicare".
#         Format your response as a HTML table with the following columns: 
#         - Measure ID
#         - Measure Name
#         - Weight,
#         - Category
#         - Subcategory
#         """
#     ],
#     huggingface_api_key=open("C:/Users/oneno/OneDrive/Desktop/HuggingFace key.txt", "r").read(),
#     document_filepath_or_url="C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/Documentation/NCQA/Measure List/Subsets/2024 Measures - Commercial.pdf",
#     query_method="map_reduce",
#     splitter_chunk_size=2000,
#     splitter_chunk_overlap=200
# )

# # Display response
# display(HTML(response))

# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.vectorstores import ElasticsearchStore
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

# support stuff
import tempfile
from icecream import ic

# import logging

# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG) # or any level you need

# # create console handler and set level to debug
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to handler
# handler.setFormatter(formatter)

# # add handler to logger
# logger.addHandler(handler)

def write_temp_file(uploaded_file):
    """
    This function writes the contents of an uploaded file to a temporary file on the system, 
    and then returns the path to that temporary file.

    Parameters:
    uploaded_file (io.BytesIO): The uploaded file as a BytesIO object.

    Returns:
    str: The path to the temporary file on the system.
    """
    # Create a new temporary file using the NamedTemporaryFile function from the tempfile module
    # The delete=False parameter means that the file won't be deleted when it's closed
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Write the contents of the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        # Save the path to the temporary file
        tmp_file_path = tmp_file.name

    # Return the path to the temporary file
    return tmp_file_path

def get_connections(provider="openai", **kwargs):
    """
    This function initializes and returns an embeddings provider and a language model 
    based on the specified provider name.

    Parameters:
    provider (str): Optional. The name of the provider to use for embeddings and language modeling. Defaults to "openai".
    kwargs (dict): Optional. Additional arguments that can be passed to the embeddings provider and language model.

    Returns:
    tuple: A tuple containing the embeddings provider and the language model.

    The function checks the provider name, and based on this, it creates and returns an instance of the OpenAIEmbeddings 
    class (for generating embeddings) and an instance of the OpenAI class (for language modeling).

    Currently, it supports only the "openai" provider, and if any other provider name is passed, it will default to "openai".

    Note: Additional parameters for the embeddings provider and language model can be passed via **kwargs, but they are not 
    used in the current implementation.
    """
    # Clean the provider name
    cleaned_provider = provider.lower().strip()

    # Check the provider name and create the appropriate embeddings provider and language model
    if cleaned_provider == "openai":
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()
    else:
        # Default to "openai" if the provider name is not recognized
        embeddings = OpenAIEmbeddings()
        llm = OpenAI()

    # Return the embeddings provider and the language model
    return embeddings, llm

def extract_text_from_upload(uploaded_file, encoding="utf8", 
                             csv_loader_args={'delimiter': ','}):
    """
    This function extracts text data from an uploaded file. The type of file - PDF, CSV, or text - 
    is determined by checking the file's type. Appropriate loaders are used to extract the text
    based on the file type.

    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user.
    encoding (str): Optional. The encoding to be used for text or CSV files. Defaults to "utf8".
    csv_loader_args (dict): Optional. Additional arguments to be passed to the CSV loader 
                            (such as the delimiter). Defaults to {'delimiter': ','}.

    Returns:
    str: The extracted text data from the file.
    """
    # Write the uploaded file to a temporary file on the system
    tmp_file_path = write_temp_file(uploaded_file)

    # Convert the file's type to lower case and remove leading/trailing whitespace
    file_type = uploaded_file.type.lower().strip()

    # Check the file type and create the appropriate loader
    if "pdf" in file_type:
        loader = PyPDFLoader(file_path=tmp_file_path)
    elif "csv" in file_type:
        loader = CSVLoader(file_path=tmp_file_path, 
                           encoding=encoding, 
                           csv_args=csv_loader_args)
    elif "text" in file_type:
        loader = TextLoader(tmp_file_path)
    else:
        # If the file type isn't recognized, assume it's a text file
        loader = TextLoader(tmp_file_path)

    # Use the loader to extract the text data from the file
    text = loader.load()

    # Return the extracted text
    return text

def load_document_text(text, 
                       es_url=None, 
                       index_name=None, 
                       embeddings=None, 
                       separator="\n", 
                       chunk_size=1000, 
                       chunk_overlap=200, 
                       length_function=len):
    """
    This function splits a given text into chunks, and then uploads these chunks to an Elasticsearch 
    cluster using an instance of the ElasticVectorSearch class.

    Parameters:
    text (str): The text to be split and uploaded.
    es_url (str): Optional. The URL of the Elasticsearch cluster. Defaults to None.
    index_name (str): Optional. The name of the Elasticsearch index where the chunks will be uploaded. Defaults to None.
    embeddings (np.ndarray or list of np.ndarray): Optional. The embeddings for the chunks, if any. Defaults to None.
    separator (str): Optional. The character used to split the text into chunks. Defaults to "\n".
    chunk_size (int): Optional. The size of each chunk. Defaults to 1000.
    chunk_overlap (int): Optional. The size of the overlap between consecutive chunks. Defaults to 200.
    length_function (function): Optional. The function used to compute the length of a chunk. Defaults to len.

    This function first creates an instance of the CharacterTextSplitter class, which splits the input text into 
    chunks. Each chunk will be of size `chunk_size`, and consecutive chunks will overlap by `chunk_overlap` characters.
    The chunks are split at the character specified by `separator`.

    The chunks are then uploaded to the Elasticsearch cluster at the specified URL, and stored in the specified index.
    If embeddings are provided, they will be associated with the chunks.
    """
    # Create an instance of CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )

    # Split the text into chunks
    chunks = text_splitter.split_documents(text)
    
    # Upload the chunks to the Elasticsearch cluster
    uploaded = ElasticsearchStore.from_documents(chunks, embeddings,
                                                  es_url=es_url, 
                                                  index_name=index_name)
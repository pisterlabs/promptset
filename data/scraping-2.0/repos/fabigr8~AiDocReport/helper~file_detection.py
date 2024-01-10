from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import configparser as cpa
import pinecone

from tqdm.auto import tqdm
from uuid import uuid4


def get_all_files(path):
    """ function to get all files from a given path
    
    Args:
        path (string): path to a folder

    Returns:
        files (list): list of all files in the folder
        
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    
    return files


def get_doc_type(path):
    """ function to detect the type of a document
    
    Args:
        path (string): path to a file

    Returns:
        file_type (string): filetype as short extionsion
        
    """
    gfile = guess(path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(gfile.extension.lower() == "pdf"):
        file_type = "pdf"
        
    elif(gfile.extension.lower() in image_types):
        file_type = "image"
        
    else:
        file_type = "unkown"
        
    return file_type


def get_file_content(file_path):
    """_summary_

    Args:
        file_path (string): path to a file

    Returns:
        documents_content (content as bytecode): content of the 
    """
    
    file_type = get_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
        
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    
    return documents_content


# Main function
def main()
    """ function to get all files from a given path and vectorize them
    
    Args:
        path (string): path to a folder with subfolders for each patient

    Returns:
        files (list): list of all files in the folder
        
    """
    # # read config file with api keys
    config = cpa.RawConfigParser()   
    configFilePath = 'config.txt'
    config.read(configFilePath)

    # # connect Pinecone
    # get API key stored in config.txt file
    PINE_API_KEY = configParser.get('PINECONE', 'PINE_API_KEY') 
    # get ENV (cloud region) stored in config.txt file
    PINE_ENV = configParser.get('PINECONE', 'PINE_ENV')
    # get index name stored in config.txt file
    PINE_INDEX = configParser.get('PINECONE', 'PINE_INDEX')
    
    # init pinecone connection
    pinecone.init(
        api_key=PINE_API_KEY,
        environment=PINE_ENV
    )

    # we create a new index if no index is found
    if index_name not in pinecone.list_indexes():   
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )
        
    # # connect to embedding model 
    # get open AI API key and defined model specified in config
    OPENAI_API_KEY = configParser.get('OPENAI', 'OPENAI_API_KEY')
    OPENAI_MODEL = configParser.get('OPENAI', 'OPENAI_MODEL')
    
    embed = OpenAIEmbeddings(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # # retrieve, vectorize and index all files
    
    files = get_all_files("documents")
    
    for file in files:
        
        file_type = get_doc_type(file)
        file_content = get_file_content(file)

        

        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        
        doc_chunks = text_splitter.split_text(file_content)
        
        for i in tqdm(range(0, len(data), batch_size)):
            # get end of batch
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]
            # first get metadata fields for this record
            metadatas = [{
                'title': record['title'],
                'text': record['context']
            } for j, record in batch.iterrows()]
            # get the list of contexts / documents
            documents = batch['context']
            # create document embeddings
            embeds = embed.embed_documents(documents)
            # get IDs
            ids = batch['id']
            # add everything to pinecone
            index.upsert(vectors=zip(ids, embeds, metadatas))
        
        # end for batch
    
    # end for file

#end main


# main callable
if __name__ == "__main__":
    main()


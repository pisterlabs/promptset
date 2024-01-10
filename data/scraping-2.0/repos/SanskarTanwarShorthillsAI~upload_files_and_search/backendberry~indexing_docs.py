from search_modules.document_loader.loader import DocumentLoaders
from search_modules.embedding_model.get_embedding import CreateEmbeddings
from search_modules.data_preparation.chunking import CreateChunks
from search_modules.index_data.indexing import Indexing
import time
import pandas as pd
from langchain.text_splitter import SpacyTextSplitter
import os
import subprocess
import concurrent.futures
import tempfile



def convert_to_pdf(input_file):
    """
    Convert supported document files to PDF using LibreOffice.

    Args:
        input_file (str): Full path of the input file.

    Returns:
        tuple: (str, str) Full path to the saved PDF file and the original file path.
    """
    try:
        # Get the file extension
        file_extension = os.path.splitext(input_file)[-1].lower()
        file_path = input_file

        # Add all the file extension present in your directory
        supported_formats = ["docx", "doc", "txt", "pptx", "docm", "xlsx", "pdf", "ppt"]

        if file_extension[1:] not in supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Command to convert the file to PDF using LibreOffice
        convert_command = [
            "libreoffice",
            "--headless",
            "--invisible",
            "--convert-to",
            "pdf",
            "--outdir",
            temp_dir,
            input_file
        ]

        # Execute the conversion command
        result = subprocess.run(convert_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Get the generated PDF file
        pdf_filename = os.path.splitext(os.path.basename(input_file))[0] + ".pdf"
        pdf_path = os.path.join(temp_dir, pdf_filename)

        return pdf_path, file_path

    except Exception as e:
        print(f"Error converting {input_file} to PDF: {str(e)}")
        error_file = "error_file_change.txt"
        with open(error_file, "a") as error_log:
            error_log.write("Filename : "+ os.path.basename(file_path) +", Exception :"+ str(e) + "\n")
        return None

def create_input_data_embeddings_for_indexing(file_path):
    """
    enter the data location and it will return  the embeddings and coressponding title,page_no,file name, text chunks &  file url/path
    vector : title + text_chunks                 
    """    
    try:
        embeddings_list=[]
        # Calling convert_to_pdf() function :
        # pdf_path,filename = convert_to_pdf(file_path)
        pdf_path=file_path
        filename=file_path
        loader=DocumentLoaders(pdf_path).getDocumentsLoader()    
        text_splitter=SpacyTextSplitter(
                chunk_size = 800,
                chunk_overlap  = 200,
                length_function = len,
                add_start_index = False,
                max_length=3000000
        )   
        text_chunks=loader.load_and_split(text_splitter=text_splitter)
        print(f"text chunks : {text_chunks}") 
        # chunk_to_be_added=chunks.getCleanData(text_chunks[0].page_content)
        chunk_to_be_added=text_chunks[0].page_content
        # page_number=text_chunks[0].metadata['page_number']
        total_token = 0
        istoaddflag=False
        for idx in range(len(text_chunks)):
            #time taken in adding a chunk
            b=time.time()
            # if chunk_to_be_added=='':
            #     # chunk_to_be_added+=" "+chunks.getCleanData(text_chunks[idx].page_content)
            #     chunk_to_be_added+=" "+text_chunks[idx].page_content

            # if istoaddflag:
            #     # chunk_to_be_added+=" "+chunks.getCleanData(text_chunks[idx].page_content)
            #     chunk_to_be_added+=" "+text_chunks[idx].page_content

            # # if not istoaddflag:
            # #     page_number=text_chunks[idx].metadata['page_number']
            
            # if chunks.getlen(chunk_to_be_added)<=200 and idx!=len(text_chunks)-1:            
            #     istoaddflag=True
            #     continue
            # if chunks.getlen(chunk_to_be_added)<=200 and idx==len(text_chunks)-1:
            #     # chunk_to_be_added=embeddings_list[len(embeddings_list)-1]['properties']['text_chunk']+" "+chunks.getCleanData(text_chunks[idx].page_content)            
            #     chunk_to_be_added=embeddings_list[len(embeddings_list)-1]['properties']['text_chunk']+" "+text_chunks[idx].page_content           
            chunk_to_be_added=text_chunks[idx].page_content
            b = time.time() - b
            data_dict=dict()
            final_data_dict=dict()
            # Updating the filepath 
            text_chunks[idx].metadata['source']=filename
            data_dict['file_path']=text_chunks[idx].metadata['source']
            # Updating the filename
            text_chunks[idx].metadata['filename']=os.path.basename(filename)
            data_dict['title']=text_chunks[idx].metadata['filename']
            data_dict['text_chunk']=chunk_to_be_added
            #Adding page_number in data_dict
            # data_dict['page_number']=page_number 
        
            final_data_dict['filename']=text_chunks[idx].metadata['filename']
            final_data_dict['properties']=data_dict
            

            #Time taken in embedding
            c = time.time()
            final_data_dict['vector']=embeddings.get_embeddings(data_dict['title']+chunk_to_be_added) #updated
            c = time.time() - c
            final_data_dict['word_count_per_chunk']=chunks.getlen(chunk_to_be_added)
            final_data_dict['time_in_chunk_added'] = str(b) 
            final_data_dict['time_in_embedding'] = str(c) 
            final_data_dict['token_length_per_chunk']=chunks.num_tokens_from_string(chunk_to_be_added,"cl100k_base")
            total_token += final_data_dict['token_length_per_chunk']
            embeddings_list.append(final_data_dict)
            # istoaddflag=False
            # chunk_to_be_added=''
        return embeddings_list,total_token

    except Exception as e:
        error_log_file="error_log.xlsx"
        # Log the error and chunk information to an Excel file
        error_data = {
            'Filename': [os.path.basename(file_path)],
            'Chunk': [text_chunks[0].page_content if text_chunks else ''],
            'Error': [str(e)]
        }
        error_df = pd.DataFrame(error_data)
        if not os.path.exists(error_log_file):
            error_df.to_excel(error_log_file, index=False)
        else:
            with pd.ExcelWriter(error_log_file, engine='openpyxl', mode='a') as writer:
                error_df.to_excel(writer, index=False, header=False)
        raise

def process_file(file_path):
    """
    Process a file for indexing.

    Args:
    - file_path (str): Path to the file for indexing.

    Embeds data and indexes to Weaviate.
    If an exception occurs, logs file name and exception details to 'error_file.txt'.

    """
    try:
        print(f"file for indexing - {file_path}")            
        data_list,total_token = create_input_data_embeddings_for_indexing(file_path)
        df = pd.DataFrame(data_list)
        df.to_excel("cost_estimation.xlsx")            
        q = time.time() - t
        print(f"time taken in embedding {q}")
        indexing.index_data_to_weaviatedb(data_list, index_name)
        s = time.time() - t
        print(f"total time taken {s}")
        print(f"total token :{total_token}")
    
    except Exception as e:
        print(e)
        # Log the file name where the error occurred
        # error_file = "error_file.txt"
        # with open(error_file, "a") as error_log:
        #      error_log.write("Filename : "+ file_path +", Exception :"+ e + "\n")

def createIndexing(directory_to_scan):
    """
    Index files in the specified directory using parallel processing.

    Args:
    - directory_to_scan (str): Path to the directory for file indexing.

    Recursively scans the directory, creates batches, and processes files in parallel
    using a ProcessPoolExecutor with 6 workers.
    """
    file_path_list = chunks.list_files_recursive(directory_to_scan)
    process_file(file_path_list[0])
    # # Create batches of files
    # batch_size = 6
    # file_batches = [file_path_list[i:i+batch_size] for i in range(0, len(file_path_list), batch_size)]
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    #     for batch in file_batches:
    #         executor.map(process_file, batch)




if __name__ == "__main__":
    t = time.time()
    index_name = "search_with_pdf"

    directory_to_scan = "upload_data"
    chunks = CreateChunks()
    indexing = Indexing()
    embeddings = CreateEmbeddings()
    # process_file('berry-free-react-admin-template/backendberry/Assignment 4 Sanskar.pdf')
    createIndexing(directory_to_scan)
    print("final time for indexing",time.time()-t)
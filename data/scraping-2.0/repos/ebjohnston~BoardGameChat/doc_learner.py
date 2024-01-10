
import pandas as pd
import json

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PagedPDFSplitter

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from langchain.document_loaders import UnstructuredWordDocumentLoader
import time
import numpy as np
import json
import concurrent.futures
import os
import openai

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

AI_LEARNED_DOCS_DIR = 'AI_learned_docs'

def get_file_type(file_path):
    return os.path.splitext(file_path)[1][1:].lower()


def process_chunk(chunk):
    embd = get_embedding(chunk, engine='text-embedding-ada-002')
    return embd




#### Loading Data

def pdf_reader_and_embeddings_generator( pdfPath ):
    filename = os.path.basename(pdfPath)
    filenamex = os.path.splitext(os.path.basename(pdfPath))[0]
    
    filesExisting = os.listdir(AI_LEARNED_DOCS_DIR)

    try:
        
        if filename.split('.')[0]+".json" in filesExisting:
            print("file_already_learened")
            return "file_already_learned"

        loader = PagedPDFSplitter(pdfPath)
        data = loader.load()
        print("typeofdata = ",type(data))
        # print("theactualdata = ",data)
        print (f'You have {len(data)} document(s) in your data')
        print (f'There are {len(data[0].page_content)} characters in your document')



        # ### Cutting into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)


        print (f'Now you have {len(texts)} documents')
        print (f'There are {len(texts[0].page_content)} characters in each of your document')

        ###### After dividing into chunks, create a json of the file. ######
        list_2d = [mytext.page_content for mytext in texts]
        df = pd.DataFrame(list_2d, columns=['chunk'])
        df.to_json(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json', orient='records')
        
        ###### Once json is written, create embedding for each chunk and write it back to json #####
        remaining_chunks = len(texts)
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json') as f:
            data = json.load(f)
            ##### Sending and calculating embeddings concurrently ######
            with concurrent.futures.ThreadPoolExecutor(5) as executor:
                futures = []
                n = 1
                for ind, i in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                    print("appending...")
                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()
                    print("future_sending...")

            # Write the updated JSON file
        print("concurrent_finished")
    
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json', 'w') as f:
            json.dump(data, f)
        
        return "Success"
    except Exception as e:
        print("An error occurred:", e)
        raise


def docx_reader_and_embeddings_generator( docxPath ):
    filename = os.path.basename(docxPath)
    filenamex = os.path.basename(docxPath).split('.')[0]
    
    filesExisting = os.listdir(AI_LEARNED_DOCS_DIR)

    try:
        
        if filename.split('.')[0]+".json" in filesExisting:
            print("file_already_learened")
            return "file_already_learned"

        # loader = PagedPDFSplitter(docxPath)
        loader = UnstructuredWordDocumentLoader(docxPath)
        data = loader.load()
        print("typeofdata = ",type(data))
        # print("theactualdata = ",data)
        print (f'You have {len(data)} document(s) in your data')
        print (f'There are {len(data[0].page_content)} characters in your document')

        # ### Cutting into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)


        print (f'Now you have {len(texts)} documents')
        print (f'There are {len(texts[0].page_content)} characters in each of your document')

        ###### After dividing into chunks, create a json of the file. ######
        list_2d = [mytext.page_content for mytext in texts]
        df = pd.DataFrame(list_2d, columns=['chunk'])
        df.to_json(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json', orient='records')
        
        ###### Once json is written, create embedding for each chunk and write it back to json #####
        remaining_chunks = len(texts)
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json') as f:
            data = json.load(f)
            ##### Sending and calculating embeddings concurrently ######
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                n=1
                for ind, i in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                    print("appending...")
                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()
                    print("future_sending...")

            # Write the updated JSON file
        print("concurrent_finished")
    
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json', 'w') as f:
            json.dump(data, f)
        
        return "Success"
    except Exception as e:
        print("An error occurred:", e)
        raise


def excel_reader_and_embeddings_generator(excelPath):
    sheet = pd.read_excel(excelPath)
    # sheet = excel_file.iloc[0]
    filename = os.path.basename(excelPath)
    # dir_path = os.path.dirname(excelPath)
    files = os.listdir(AI_LEARNED_DOCS_DIR)
    print("thefilename: ",filename)
    if filename.split('.')[0]+'.json' in files:
        print('excel {} already learned.'.format(filename))
        return 'excel {} already learned.'.format(filename)
    try:
        # Create an empty list to store the rows as strings
        rows_list = []

        # Loop through each row in the sheet
        for i, row in sheet.iterrows():
            # Concatenate the column name and value into a single string
            row_str = ''
            for col_name in row.index:
                col_name_str = str(col_name)
                col_value_str = str(row[col_name])
                row_str += col_name_str + ': ' + col_value_str + ' ' +'-> '

            # Check the length of the string and append it to the list if it's less than or equal to 3000 characters
            if len(row_str) <= 10800:
                rows_list.append({"chunk":row_str})
            else:
                print("skipped because length = ",len(row_str),flush=True)

        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json', 'w') as f:
                    json.dump(rows_list, f)


        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json') as f:
            data = json.load(f)
            ##### Sending and calculating embeddings concurrently ######
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                n=1
                for ind, i in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    # time.sleep(0.2)
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                    print("appending...",flush=True)
                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()
                    print("future_sending...",flush=True)

            # Write the updated JSON file
        print("concurrent_finished")

        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json', 'w') as f:
            json.dump(data, f)

        print("completedd")
        return "Success"
    except Exception as e:
        print("An error occurred:", e)
        raise


def csv_reader_and_embeddings_generator(excelPath):
    sheet = pd.read_csv(excelPath)
    # sheet = excel_file.iloc[0]
    filename = os.path.basename(excelPath)
    # dir_path = os.path.dirname(excelPath)
    files = os.listdir(AI_LEARNED_DOCS_DIR)
    print("thefilename: ",filename)
    if filename.split('.')[0]+'.json' in files:
        print('excel {} already learned.'.format(filename))
        return 'excel {} already learned.'.format(filename)
    try:
        # Create an empty list to store the rows as strings
        rows_list = []

        # Loop through each row in the sheet
        for i, row in sheet.iterrows():
            # Concatenate the column name and value into a single string
            row_str = ''
            for col_name in row.index:
                col_name_str = str(col_name)
                col_value_str = str(row[col_name])
                row_str += col_name_str + ': ' + col_value_str + ' ' +'-> '

            # Check the length of the string and append it to the list if it's less than or equal to 3000 characters
            if len(row_str) <= 3000:
                rows_list.append({"chunk":row_str})

        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json', 'w') as f:
                    json.dump(rows_list, f)


        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json') as f:
            data = json.load(f)
            ##### Sending and calculating embeddings concurrently ######
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                n=1
                for ind, i in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                    print("appending...")

                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()
                    print("future_sending...")

            # Write the updated JSON file
        print("concurrent_finished")

        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json', 'w') as f:
            json.dump(data, f)

        print("completedd")
    except Exception as e:
        print("An error occurred:", e)
        raise



def txt_reader_and_embeddings_generator( txtPath ):
    filename = os.path.basename(txtPath)
    filenamex = os.path.basename(txtPath).split('.')[0]
    
    filesExisting = os.listdir(AI_LEARNED_DOCS_DIR)

    try:
        if filename.split('.')[0]+".json" in filesExisting:
            print("file_already_learened: ",filename)
            return "file_already_learned: "+str(filename)
        
        with open(txtPath, 'r', encoding='utf-8') as f:
            file_contents = f.read()

        chunk_list = []
        chunk = ''
        for char in file_contents:
            if len(chunk) < 1000:
                chunk += char
            else:
                chunk_list.append({"chunk":chunk})
                chunk = char

        if chunk:
            chunk_list.append({"chunk":chunk})

        with open(AI_LEARNED_DOCS_DIR+'/'+filename.split('.')[0]+'.json', 'w') as f:
            json.dump(chunk_list, f)

        
        
        ###### Once json is written, create embedding for each chunk and write it back to json #####
        
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json') as f:
            data = json.load(f)
            ##### Sending and calculating embeddings concurrently ######
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                n=1
                for ind, (i) in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                    print("appending...")
                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()
                    print("future_sending...")

            # Write the updated JSON file
        print("concurrent_finished")
    
        with open(f'{AI_LEARNED_DOCS_DIR}/{filenamex}.json', 'w') as f:
            json.dump(data, f)
        
        return "Success"
    except Exception as e:
        print("An error occurred:", e)
        raise

def read_file_and_generate_embeddings(file_path):
    """
    Reads a file and generates embeddings for its contents.

    Args:
    file_path: A string representing the path to the input file.

    Returns:
    A string indicating whether the file was successfully processed or if it was already learned.
    """
    # Check if the file has already been learned
    filename = os.path.basename(file_path)
    files = os.listdir(AI_LEARNED_DOCS_DIR)
    if filename.split('.')[0] + '.json' in files:
        print('File {} has already been learned.'.format(filename))
        return 'File {} has already been learned.'.format(filename)

    try:
        # Read the file
        if file_path.endswith('.xlsx'):
            sheet = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            sheet = pd.read_csv(file_path)
        else:
            raise ValueError('Invalid file format')

        # Create an empty list to store the rows as strings
        rows_list = []

        # Loop through each row in the sheet
        for i, row in sheet.iterrows():
            # Concatenate the column name and value into a single string
            row_str = ''
            for col_name in row.index:
                col_name_str = str(col_name)
                col_value_str = str(row[col_name])
                row_str += col_name_str + ': ' + col_value_str + ' ' +'-> '

            # Check the length of the string and append it to the list if it's less than or equal to 3000 characters
            if len(row_str) <= 3000:
                rows_list.append({"chunk":row_str})

        # Write the rows as JSON to disk
        with open(os.path.join(AI_LEARNED_DOCS_DIR, filename.split('.')[0]+'.json'), 'w') as f:
            json.dump(rows_list, f)

        # Load the JSON data and generate embeddings concurrently
        with open(os.path.join(AI_LEARNED_DOCS_DIR, filename.split('.')[0]+'.json')) as f:
            data = json.load(f)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                n=1
                for ind, i in enumerate(data):
                    future = executor.submit(process_chunk, i["chunk"])
                    futures.append((ind, future))
                    n+=1
                    if n>=500:
                        n=1
                        time.sleep(14)
                # Update the JSON file with the embeddings
                for ind, future in futures:
                    data[ind]["embeddings"] = future.result()

        # Write the updated JSON file to disk
        with open(os.path.join(AI_LEARNED_DOCS_DIR, filename.split('.')[0]+'.json'), 'w') as f:
            json.dump(data, f)

        print("File {} processed successfully.".format(filename))
        return "File {} processed successfully.".format(filename)

    except Exception as e:
        print("An error occurred:", e)
        raise


def delete_file_from_both_folders(file_name):
    # Delete file from AI_DOCS directory
    ai_docs_path = f'./{AI_LEARNED_DOCS_DIR}'
    ai_file_path = os.path.join(ai_docs_path, file_name + ".json")
    try:
        os.remove(ai_file_path)
        print(f"File '{file_name}' deleted successfully from AI_DOCS.")
    except OSError as e:
        print(f"Error deleting file '{file_name}' from AI_DOCS: {e}")
    
    # Delete file from RAW_DOCS directory
    raw_docs_path = "./RAW_DOCS"
    for ext in ["xlsx", "pdf", "txt", "docx", "csv"]:
        raw_file_path = os.path.join(raw_docs_path, file_name + "." + ext)
        if os.path.exists(raw_file_path):
            try:
                os.remove(raw_file_path)
                print(f"File '{file_name}' deleted successfully from RAW_DOCS.")
                break  # exit loop if file is found and deleted
            except OSError as e:
                print(f"Error deleting file '{file_name}' from RAW_DOCS: {e}")
    
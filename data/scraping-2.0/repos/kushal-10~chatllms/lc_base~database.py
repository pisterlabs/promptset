from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 

import os 
import shutil

import pandas as pd

class Data():
    def __init__(self, inp_dir='reports', out_dir="output_reports") -> None:
        self.data_dir = inp_dir
        self.out_dir = out_dir
        pass

    def check_output(self):
        '''
        Create an output folder to save texts of individual PDFs
        Remove folder if it exists and create new
        '''
        folder_path = self.out_dir
        # Check if the folder exists
        if os.path.exists(folder_path):
            # If the folder exists, delete its content
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print("Folder content deleted.")
        else:
            # If the folder doesn't exist, create it
            try:
                os.makedirs(folder_path)
                print("Folder created.")
            except Exception as e:
                print(f"Failed to create folder. Reason: {e}")



    def get_faiss_embeddings(self):
        '''
        Splits all the documents, saves them in text format
        '''
        # Get a list of all PDFs in the specified directory
        list_pdfs = os.listdir(self.data_dir)
        # Initialize OPENAI embeddings
        embedding = OpenAIEmbeddings()

        # Make directories for each pdf separately 
        pdf_names = []
        pdf_num = []
        dir_num = 0
        text_count = 0
        for pdf in list_pdfs:
            dir_num += 1
            new_dir = os.path.join(self.out_dir, str(dir_num))
            os.makedirs(new_dir)
            print('Creating Database for PDF ' + str(dir_num))
            pdf_file = os.path.join(self.data_dir, pdf)
            reader = PdfReader(pdf_file)

            # Get the textual content of PDF
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            # Split the texts 
            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200, 
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)
            text_count += len(raw_text)
            print('Length of text: ' + str(len(raw_text)))
            # Create Embedding
            db = FAISS.from_texts(texts, embedding)

            # Save Embedding
            db.save_local(os.path.join(new_dir, "faiss_index"))

            pdf_names.append(pdf)
            pdf_num.append(dir_num)
        
        data_df = {
            "names": pdf_names,
            "index": pdf_num
        }
        df = pd.DataFrame(data_df)
        map_name = os.path.split(self.out_dir)[-1]
        df.to_csv(os.path.join("outputs", "mappings", str(map_name) + ".csv"))
        print('Total text in data: ' + str(text_count))

        return None

    def get_combined_faiss_embedding(self):
        '''
        Combines all the documents, saves them in ChromaDB format
        '''
        # Get a list of all PDFs in the specified directory
        list_pdfs = os.listdir(self.data_dir)
        # Initialize OPENAI embeddings
        embedding = OpenAIEmbeddings()

        raw_text = ''
        for pdf in list_pdfs:
            print('Creating Database for PDF ' + str(pdf))
            pdf_file = os.path.join(self.data_dir, pdf)
            reader = PdfReader(pdf_file)

            # Get the textual content of PDF
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            # Split the texts 
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200, 
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        text_count = len(raw_text)
        print('Length of text: ' + str(len(raw_text)))
        # Create Embedding
        db = FAISS.from_texts(texts, embedding)

        # Save Embedding
        db.save_local(os.path.join(self.out_dir, "faiss_index"))

        print('Total text in data: ' + str(text_count))



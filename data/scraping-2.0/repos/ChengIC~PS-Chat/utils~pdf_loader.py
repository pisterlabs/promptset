from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain.document_loaders import TextLoader
from cleantext import clean
import re
from pdf2image import convert_from_path
from pytesseract import image_to_string
import time
import logging

def pdf_to_text(file_path):
    images = convert_from_path(file_path)  # Convert the PDF into images
    full_text = ""
    for i, image in enumerate(images):
        text = image_to_string(image)  # Use OCR to convert the image to text
        full_text += text
    return full_text

def get_words_from_pdf(file_path):
    # Open the PDF file in read-binary mode
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf = PdfReader(file)
        # Initialize an empty string for text data from PDF
        text = ''
        # Iterate over the number of pages in the PDF
        for page in pdf.pages:
            # Add each page's text to the overall text string
            text += page.extract_text()
        # Return the words from the PDF as a list
        return text.split()

def find_smallest_pdf_and_count_words(directory_path):
    # Initialize the smallest file size and name as None
    smallest_file_size = None
    smallest_file_name = None

    # Iterate over the filenames in the given directory
    for filename in os.listdir(directory_path):
        # If this file is a PDF
        if filename.endswith('.pdf'):
            # Get the file size
            file_size = os.path.getsize(os.path.join(directory_path, filename))
            # If this file is smaller than the current smallest file, or if there is no smallest file yet
            if smallest_file_size is None or file_size < smallest_file_size:
                # Set this file as the new smallest file
                smallest_file_size = file_size
                smallest_file_name = filename

    # Initialize the word count as None
    word_count = None

    if smallest_file_name is not None:
        # Get the words from the smallest PDF
        words = get_words_from_pdf(os.path.join(directory_path, smallest_file_name))
        # Count the words
        word_count = len(words)

    # Return the smallest PDF's filename and word count
    return smallest_file_name, word_count


def get_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolder = os.path.join(root, dir)
            subfolders.append(subfolder)
    return subfolders

class ingest_pdf:
    def __init__(self, namespace, pinecone_api_key, pinecone_env_name, pinecone_index_name, 
                doc_dir):
        
        self.namespace=namespace
        self.pinecone_api_key=pinecone_api_key
        self.pinecone_env_name=pinecone_env_name
        self.pinecone_index_name=pinecone_index_name
        self.doc_dir=doc_dir

        # initialize pinecone
        self.pinecone = pinecone.init(
                        api_key=pinecone_api_key,  # find at app.pinecone.io
                        environment=pinecone_env_name  # next to api key in console
                    )

    def check_folders_pdf_info(self):

        subfolders = get_subfolders(self.doc_dir)
        for subfolder in tqdm(subfolders):
            print (subfolder, find_smallest_pdf_and_count_words(subfolder))


    def check_pinecone_index(self):
        print ("Checking if pinecone index exists...")
        index = pinecone.Index(self.pinecone_index_name)
        print(index.describe_index_stats())


    def upload_pdf_to_pinecone(self, chunk_size=1000, chunk_overlap=0):

        print ("Start uploading PDFs to Pinecone...")
        
        index = pinecone.Index(self.pinecone_index_name)
        available_namesacpes = index.describe_index_stats()["namespaces"]

        subfolders = get_subfolders(self.doc_dir)
        if len(subfolders)>=1:
            for subfolder in tqdm(subfolders):
                my_namespace = subfolder.split(self.doc_dir)[1].replace("/","")

                if my_namespace in available_namesacpes:
                    print ("Namespace " + my_namespace + " already exists in pinecone index. Skipping...")
                    continue
                
                for filename in os.listdir(subfolder):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(subfolder, filename)
                        print (f"start splitting document{file_path}...")
                        my_loader = PyPDFLoader(file_path)
                        documents = my_loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
                        docs = text_splitter.split_documents(documents)

                        # dealing with unstructured pdf
                        if len(docs)==0:
                            print (f"start splitting document{file_path} with unstructured mode")
                            text = pdf_to_text(file_path)
                            converted_file_path = os.path.join(subfolder, f'{filename.split(".pdf")[0]}_converted.txt')
                            with open(converted_file_path, 'w') as file:
                                file.write(text)
                            my_loader = TextLoader(converted_file_path)
                            documents = my_loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
                            docs = text_splitter.split_documents(documents)
                            print(len(docs))

                        print (f"finsihed splitting document{file_path} and bigin uploading...")
                        
                        Pinecone.from_documents(docs, OpenAIEmbeddings(), 
                                                            index_name=self.pinecone_index_name, 
                                                            namespace=my_namespace)
                        print (f"finsihed upload document{file_path} to pinecone...")

                print ("Finished uploading PDFs to Pinecone in folder " + subfolder + " with namespace " + my_namespace)
        
        else:
            print ("No subfolders found in " + self.doc_dir)
            # upload all documents in the directory
            my_namespace = self.namespace
            if my_namespace== None:
                my_namespace = "MyPDFs"
            my_loader = DirectoryLoader(self.doc_dir, glob='**/*.pdf')
            documents = my_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            docs = text_splitter.split_documents(documents)
            
            Pinecone.from_documents(docs, OpenAIEmbeddings(), 
                                                index_name=self.pinecone_index_name, 
                                                namespace=my_namespace)
            print ("Finished uploading PDFs to Pinecone in folder " + subfolder + " with namespace " + my_namespace)
        
    def delete_namespaces (self, namespace): 
        print ("Deleting namespaces...")
        index = pinecone.Index(self.pinecone_index_name)
        index.delete(delete_all=True, namespace=namespace)
        print (f"Finished deleting namespaces.{namespace}")

    def delete_all_namespaces (self): 
        index = pinecone.Index(self.pinecone_index_name)
        available_namesacpes = index.describe_index_stats()["namespaces"]
        for namespace in available_namesacpes:
            index.delete(delete_all=True, namespace=namespace)
            print (f"Finished deleting namespaces.{namespace}")
        print ("Finished deleting all namespaces.")
    
    def get_namespaces(self):
        index = pinecone.Index(self.pinecone_index_name)
        available_namesacpes = index.describe_index_stats()["namespaces"]
        return available_namesacpes

    def upload_pdf_to_pinecone_v2(self, chunk_size=1000, chunk_overlap=0, pdf_namespace="my-pdf"):
        print ("Start uploading PDFs to Pinecone...")
        index = pinecone.Index(self.pinecone_index_name)
        subfolders = get_subfolders(self.doc_dir)
        for subfolder in tqdm(subfolders):
            my_namespace = pdf_namespace
            
            for filename in os.listdir(subfolder):
                
                if filename.endswith('.pdf'):
                    file_path = os.path.join(subfolder, filename)
                    print (f"start splitting document{file_path}...")
                    my_loader = PyPDFLoader(file_path)
                    documents = my_loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
                    docs = text_splitter.split_documents(documents)

                    # dealing with unstructured pdf
                    if len(docs)==0:
                        print (f"start splitting document{file_path} with unstructured mode")
                        text = pdf_to_text(file_path)
                        converted_file_path = os.path.join(subfolder, f'{filename.split(".pdf")[0]}_converted.txt')
                        with open(converted_file_path, 'w') as file:
                            file.write(text)
                        my_loader = TextLoader(converted_file_path)
                        documents = my_loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
                        docs = text_splitter.split_documents(documents)

                    print (f"finished splitting document{file_path} and bigin uploading...")
                    for doc in docs:
                        if len(doc.page_content)<50:
                            continue
                        print ('length of orginal chunk:',len(doc.page_content))
                        doc.page_content = clean(doc.page_content,
                                                fix_unicode=True,               # fix various unicode errors
                                                to_ascii=True,                  # transliterate to closest ASCII representation
                                                lower=True,                     # lowercase text
                                                no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                                                no_urls=False,                  # replace all URLs with a special token
                                                no_emails=False,                # replace all email addresses with a special token
                                                no_phone_numbers=False,         # replace all phone numbers with a special token
                                                no_numbers=False,               # replace all numbers with a special token
                                                no_digits=True,                # replace all digits with a special token
                                                no_currency_symbols=False,      # replace all currency symbols with a special token
                                                no_punct=False,                 # remove punctuations
                                                lang="en"                       # set to 'de' for German special handling
                                            )
                        def remove_unwanted_text(text):
                            # Regular expression pattern to match the unwanted text
                            pattern = r"deutsch \(\/\?lang=de_de\) \| english \| espafiol \(\/\?lang=es es\) \| francais \(\/2lang=fr_fr\) \| bahasa indonesia \(\/\?lang=id_id\) \| italiano \(\/\?lang=it_it\) \|\nnederlands \(\/\?lang=nl_nl\) \| portugues \(br\) \(\/\?lang=pt_br\) \| \"3% \(cn\) \(\/\?lang=zh_cn\)\ntry maturity 360 \(\/cmsa\/try\) \| help \(\/en\/help\) \| register free \(\/register\) \| login \(existing _users\) \(\/login\)\nld\.\nq "
                            # Replace the matched pattern with an empty string
                            cleaned_text = re.sub(pattern, '', text)
                            return cleaned_text

                        doc.page_content = remove_unwanted_text(doc.page_content)
                        print ('length of filtered chunks:', len(doc.page_content))
                    

                    success = False
                    attempt = 0
                    max_retries = 5
                    retry_delay = 30
                    while not success and attempt < max_retries:
                        try:
                            Pinecone.from_documents(docs, OpenAIEmbeddings(), 
                                                    index_name=self.pinecone_index_name, 
                                                    namespace=my_namespace)
                            print(f"Finished upload document{file_path} to Pinecone...")
                            success = True

                        except Exception as e:
                            logging.error(f"Error uploading document {file_path}: {e}")
                            attempt += 1
                            time.sleep(retry_delay)
                            print(f"Retrying upload ({attempt}/{max_retries})...")

                    if not success:
                        print(f"Failed to upload document {file_path} after {max_retries} attempts.")

            print ("Finished uploading PDFs to Pinecone in folder " + subfolder + " with namespace " + my_namespace)



#!/usr/bin/env python
# coding: utf-8

# All imports

import fitz
import PyPDF2
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
# from langchain.docstore.document import Document
from llama_index.readers.schema.base import Document
import pandas as pd
import streamlit as st
from io import BytesIO
import os
import time
import re
import warnings

warnings.filterwarnings("ignore")

# import camelot
# import PyPDF2
# from PyPDF2 import PdfReader
# from typing import Any, List, Optional
# from pathlib import Path

####

def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a string
    possibly with whitespace in between
    """
    return re.sub(r"\s*\n\s*", "\n", text)


# Function to read documents from uploaded files 
@st.cache_data(show_spinner = False)
def read_documents_from_uploaded_files(uploaded_files):
    '''    
    read_documents_from_uploaded_files:
    Explanation: Reads the documents that were uploaded

    Input - 
    uploaded_files: The uploaded files in the st.file_uploader widget

    Output -
    docs: list of documents (contains data of each of the documents)
    '''

    reader = PyMuPDFReader()

    # Initialize docs as an empty list (contains data of each of the documents)
    docs = []


    for uploaded_file in uploaded_files:

    # st.write(type(uploaded_file))
    # st.write(dir(uploaded_file))
    # st.write("Above was for pdf")

        if uploaded_file.name.endswith(".pdf"):

            # Open pdf file
            pdf = fitz.open(stream = uploaded_file.read(), filetype="pdf")

            # Got below code from Llamahub PyMuPdf base.py file - Essentially replacing this code: docs.append(reader.load(file_path))
            extra_info = {}
            extra_info["total_pages"] = len(pdf)
            extra_info["file_path"] = uploaded_file.name # Added this line of code myself

            docs.append([
                Document(
                    text=page.get_text().encode("utf-8"),
                    extra_info=dict(
                        extra_info,
                        **{
                            "source": f"{page.number+1}"
                        },
                    ),
                )
                for page in pdf
            ])

            # pdf.close()

        # Resetting stream position - so we can use fitz.open() in another instance
        uploaded_file.seek(0)
        # uploaded_file.seek(0,0)

    return docs


@st.cache_data(show_spinner = False)
def display_document_from_uploaded_files(uploaded_files):
    '''
    display_document_from_uploaded_files:
    Explanation: Displays the pdf documents that were uploaded

    Input - 
    uploaded_files: The uploaded files in the st.file_uploader widget

    Output -
    content: The entire content fo the document (with page number)
    content_document_list: List containing content of each document in string
    content_filename: List containing filename of each of the document in string
    '''

    # List containing filename of each of the document in string
    content_filename = []

    # List containing content of each document in string
    content_document_list = []
    

    for uploaded_file in uploaded_files:

        # List containing content of each page in string - Needs to get reset in each iteration
        # content_page_list = []

        # Append filename to list
        content_filename.append(uploaded_file.name)

        # Content string
        content = ""

        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(stream = uploaded_file.read(), filetype="pdf")

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            content += f"### Page number {page_number} \n"
            content += page.get_text()
            # content_page_list.append(page.get_text())

        # Append content of each document to list
        content_document_list.append(content)

        # Close the PDF file
        pdf_document.close()

        # Resetting stream position - so we can use fitz.open() in another instance
        uploaded_file.seek(0)

    return content, content_document_list, content_filename



# Function to read documents passed to it 
@st.cache_data(show_spinner = False)
def read_documents_from_uploaded_files_old(uploaded_files):
    '''
    Deprecated function - Not used anymore in code
    '''

    # Initialize docs for all document an empty list (this will be a list containing multiple list for each documents content)
    docs_for_all_document = []

    for uploaded_file in uploaded_files:

        # st.write(type(uploaded_file))
        # st.write(dir(uploaded_file))
        # st.write("Above was for pdf")

        if uploaded_file.name.endswith(".pdf"):

            # Open pdf file
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

            # Initialize docs as an empty list (contains data of each of the documents)
            docs_for_one_document = []

            # Iterates over each page of the PDF
            for i, page in enumerate(pdf):

                # Extracts the text from the page (in reading order)
                text = page.get_text(sort=True)

                # Removes consecutive newlines and extra whitespace
                text = strip_consecutive_newlines(text)

                # Creates a new Document object using the cleaned text content as the page content
                doc = Document(page_content=text.strip())

                # Sets the page number
                doc.metadata["page"] = i + 1

                # Sets the source of the document
                doc.metadata["source"] = f"p-{i+1}"

                # Appends the Document object to the list
                docs_for_one_document.append(doc)

            # Appends the docs_for_one_document list
            docs_for_all_document.append(docs_for_one_document)

            # file.read() mutates the file object, which can affect caching
            # so we need to reset the file pointer to the beginning
            uploaded_file.seek(0)

    # Return list of documents (contains data of each of the documents)
    return docs_for_all_document


# Function to save uploaded file to directory
@st.cache_resource(show_spinner = False)
def save_uploaded_file(uploadedFiles):
    '''
    save_uploaded_file:
    Explanation: Takes the uploadedFiles and writes it to Document directory 

    Input - 
    uploadedFile: A list of uploaded files (from st.file_uploader)

    Output -
    None
    '''
    for uploaded_file in uploadedFiles:

        if uploaded_file.name.endswith(".pdf"):

            # Open image file with name as writebyte and then write using buffer (in same level as directory)
            with open(os.path.join("Documents", uploaded_file.name), "wb") as f:

                f.write(uploaded_file.getbuffer())

            # Return success message
            # return st.success("Saved file :{} in Documents folder".format(uploaded_file.name))


# Function to read documents in particular directory 
@st.cache_data(show_spinner = False)
def read_documents_from_directory(directory):
    '''
    Deprecated, not used anymore as we read directly from file uploader than particular directory (this approach worked for localhost)
    read_documents_from_directory:
    Explanation: Reads the pdf documents in a particular directory

    Input - 
    directory: the directory/folder of where the pdf documents are located. Default value is "Documents"

    Output -
    docs: list of documents (contains data of each of the documents)
    '''

    reader = PyMuPDFReader()

    # Initialize docs as an empty list (contains data of each of the documents)
    docs = []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            
            # Load the file and append the data to docs
            docs.append(reader.load(file_path))

    # Return list of documents (contains data of each of the documents)
    return docs



# Function to get tables from uploaded files 
@st.cache_data(show_spinner = False)
def get_tables_from_uploaded_file(uploaded_files):
    '''
    get_tables_from_uploaded_file:  
    Explanation: Obtain tables from uploaded_files

    Input - 
    uploaded_files: List of uploaded files

    Output - 
    table_dfs: list containing dataframe of various tables
    '''

    # Dataframe of the tables from all documents we will extract
    table_dfs = []


    for uploaded_file in uploaded_files:

        if uploaded_file.name.endswith(".pdf"):

            # st.write(uploaded_file)
            # st.write(type(uploaded_file))
            # st.write("---")
            # st.write(uploaded_file.read())

            doc = fitz.open(stream = uploaded_file.read(), filetype="pdf")

            num_pages = doc.page_count        

            # Loop through all pages
            for page in range(0, num_pages):
                
                start_time = time.time()
                current_page = doc[page]
                
                # Read pdf to extract tables from that specific page
                table_object = current_page.find_tables()
                
                # This uses the tables attribute of object named TableFinder - Gives list of tables
                table_list = table_object.tables
                
                non_empty_table_counter = 0
                empty_table_counter = 0
                
                # If table_list is empty
                if len(table_list) == 0:
                    pass

                else:            
                    
                    for table in table_list:
                        if table.to_pandas().empty:

                            empty_table_counter += 1

                            elapsed_time_for_empty_table = time.time() - start_time

                        else:

                            non_empty_table_counter += 1

                            table_df = table.to_pandas()
                            
                            table_df = (
                                table_df.rename(columns=table_df.iloc[0])
                                .drop(table_df.index[0])
                                .reset_index(drop=True)
                            )

                            # Append to list
                            table_dfs.append(table_df)

                            elapsed_time_for_table = time.time() - start_time
    
    # return table_dfs (list containing dataframe of various tables)
    return table_dfs



# Function to get tables from particular directory 
@st.cache_data(show_spinner = False)
def get_tables_from_directory(path: str):

    '''
    Deprecated function (this works for localhost mainly)
    '''

    # Dataframe of the tables from the document we will extract
    table_dfs = []

    with open(path, 'rb') as file:
        doc = fitz.open(path)
        num_pages = doc.page_count

    # Loop through all pages
    for page in range(0, num_pages):
        
        start_time = time.time()
                
        current_page = doc[page]
        
        # Read pdf to extract tables from that specific page
        table_object = current_page.find_tables()
        
        # This uses the tables attribute of object named TableFinder - Gives list of tables
        table_list = table_object.tables
        
        non_empty_table_counter = 0
        empty_table_counter = 0
        
        # If table_list is empty
        if len(table_list) == 0:
            pass

        else:            
            
            for table in table_list:
                if table.to_pandas().empty:

                    empty_table_counter += 1

                    elapsed_time_for_empty_table = time.time() - start_time

                else:

                    non_empty_table_counter += 1

                    table_df = table.to_pandas()
                    
                    table_df = (
                        table_df.rename(columns=table_df.iloc[0])
                        .drop(table_df.index[0])
                        .reset_index(drop=True)
                    )

                    # Append to list
                    table_dfs.append(table_df)

                    elapsed_time_for_table = time.time() - start_time
    
    # return table_dfs (list containing dataframe of various tables)
    return table_dfs


# iterate over files in that directory & remove all "\n" characters from the dataframe
# Note this is where the above function (get_tables is called)
@st.cache_data(show_spinner = False)
def iterate_files_from_directory(directory):
    '''
    Deprecated function (this works for localhost mainly)
    '''

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            
            # Call get_tables function
            table_dfs = get_tables_from_directory(file_path)


    # Remove all "\n" characters from dataframe
    for i in range(len(table_dfs)):
        table_dfs[i] = table_dfs[i].replace('\n','', regex=True)

    # return table_dfs (list containing dataframe of various tables)
    return table_dfs

@st.cache_data(show_spinner = False)
def iterate_files_from_uploaded_files(uploaded_files):
    '''
    iterate_files_directory:  
    Explanation: iterate over files in that directory & remove all "\n" characters from the dataframe
    Explanation (contd): This is where the above function (get_tables) is called

    Input - 
    uploaded_files: Uploaded files from st.file_uploader

    Output - 
    table_dfs: list containing dataframe of various tables
    '''
    
    # Call get_tables_from_directory function
    table_dfs = get_tables_from_uploaded_file(uploaded_files)

    # Remove all "\n" characters from dataframe
    for i in range(len(table_dfs)):
        table_dfs[i] = table_dfs[i].replace('\n','', regex=True)

    # return table_dfs (list containing dataframe of various tables)
    return table_dfs

@st.cache_data(show_spinner = False)
def print_file_details(uploadedFiles):
    '''
    print_file_details:
    Explanation: Diagnostic function that displays file name, type and size 

    Input - 
    uploadedFiles: A list of uploaded files (from st.file_uploader)

    Output -
    Writing to streamlit
    '''

    # Obtain information about the file
    for file in uploadedFiles:

        # Printing file details - diagnostic purposes
        file_details = {
            "filename": file.name,
            "filetype": file.type,
            "filesize": file.size,
        }

        # Display on streamlit
        st.write(file_details)

        st.write(type(file))



@st.cache_data(show_spinner = False)
def iterate_excel_files_from_directory(excel_directory, input_excel_file, file_extension):
    '''
    iterate_excel_files_from_directory: This function ultimately reads the excel file from directory

    Input - 
    excel_directory: Directory storing excel files
    input_excel_file: Excel file name
    file_extension: Extension of excel file

    Output - 
    excel_file: The dataframe of the first sheet in the results excel file
    info_excel_file: The dataframe of sheet name, "Info", in the results excel file
    '''

    for filename in os.listdir(excel_directory):

        if filename.endswith(f".{file_extension}"):

            name_without_extension = os.path.splitext(filename)[0]

            if name_without_extension == "results":
                
                file_path = os.path.join(excel_directory, filename)
                
                excel_file = pd.read_excel(f"{excel_directory}/{input_excel_file}.{file_extension}")
                
                info_excel_file = pd.read_excel(f"{excel_directory}/{input_excel_file}.{file_extension}", sheet_name='Info')

                excel_file.dropna(axis=1, how='all', inplace=True)

                info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


    # return the first sheet of the excel file and info sheet of excel file as pandas dataframe 
    return excel_file, info_excel_file

@st.cache_data(show_spinner = False)
def iterate_uploaded_excel_file(uploaded_file):
    '''
    iterate_uploaded_excel_file: This function ultimately reads the uploaded excel file

    Input - 
    uploaded_file: Excel files uploaded using st.file_uploader

    Output - 
    excel_file: The dataframe of the first sheet in the results excel file
    info_excel_file: The dataframe of sheet name, "Info", in the results excel file
    '''
    excel_file = None
    info_excel_file = None

    if uploaded_file:
    
        # st.write(type(uploaded_file))
        # st.write(dir(uploaded_file))
        # st.write(uploaded_file.name)
        # st.write("Above was for excel")

        if uploaded_file.name.endswith(".xlsx"):

            excel_file = pd.read_excel(uploaded_file, engine='openpyxl')

            info_excel_file = pd.read_excel(uploaded_file, sheet_name='Info')

            excel_file.dropna(axis=1, how='all', inplace=True)

            info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


        elif uploaded_file.name.endswith(".xls"):

            excel_file = pd.read_excel(uploaded_file)
            
            info_excel_file = pd.read_excel(uploaded_file, sheet_name='Info')

            excel_file.dropna(axis=1, how='all', inplace=True)

            info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


    # return the first sheet of the excel file and info sheet of excel file as pandas dataframe 
    return excel_file, info_excel_file


@st.cache_data(show_spinner = False)
def iterate_uploaded_excel_files(uploadedFiles):
    '''
    iterate_uploaded_excel_files: This function ultimately reads the uploaded excel files

    Input - 
    uploadedFiles: Excel files uploaded using st.file_uploader

    Output - 
    excel_file: The dataframe of the first sheet in the results excel file
    info_excel_file: The dataframe of sheet name, "Info", in the results excel file
    '''
    excel_file = None
    info_excel_file = None

    for uploaded_file in uploadedFiles:

        st.write(type(uploaded_file))
        st.write(dir(uploaded_file))
        st.write(uploaded_file.name)
        st.write("Above was for excel")

        if uploaded_file.name.endswith(".xlsx"):

            excel_file = pd.read_excel(uploaded_file, engine='openpyxl')

            info_excel_file = pd.read_excel(uploaded_file, sheet_name='Info')

            excel_file.dropna(axis=1, how='all', inplace=True)

            info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


        elif uploaded_file.name.endswith(".xls"):

            excel_file = pd.read_excel(uploaded_file)
            
            info_excel_file = pd.read_excel(uploaded_file, sheet_name='Info')

            excel_file.dropna(axis=1, how='all', inplace=True)

            info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


    # return the first sheet of the excel file and info sheet of excel file as pandas dataframe 
    return excel_file, info_excel_file


@st.cache_data(show_spinner = False)
def show_dataframes(df_list):
    '''
    show_dataframes: This function ultimately shows the dataframe in a list format

    Input - 
    df: List of dataframes

    Output - 
    None
    '''
    for df in df_list:
        st.dataframe(data = df, use_container_width = True, column_order = None)
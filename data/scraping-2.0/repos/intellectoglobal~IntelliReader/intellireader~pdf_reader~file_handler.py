from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from pdf_reader.Error_handling import EmptyFile
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader

class FileHandler:
        
        def __init__(self, file):
            self.file = file

        def extract_pdf_content_to_document(self):
                file = self.file
                try:
                    with open(f"./input/{file.name}", 'wb') as f:
                        f.write(file.read())
                        name=file.name
                        loader = PyPDFLoader(f'./input/{name}')
                        documents = loader.load()
                        if not documents:
                            raise EmptyFile("No documents found in the Excel file.")
                except FileNotFoundError as e:
                    st.error(f"File not found: {e}")
                    return None
                except Exception as e:
                    st.error(f"An error occurred on extract pdf: {str(e)}")
                    return None
                # print(documents)
                return documents
        
        def extract_word_content_to_document(self):
            try:
                with open(f"./input/{self.file.name}", 'wb') as f:
                    f.write(self.file.read())
                    name = self.file.name
                    loader = Docx2txtLoader(f'./input/{name}')
                    documents = loader.load()
                    if not documents:
                        raise EmptyFile("No documents found in the Excel file.")
            except FileNotFoundError as e:
                st.error(f"File not found: {e}")
            except Exception as e:
                st.error(f"An error occurred on extract Word file: {str(e)}")
                return None
            return documents
        
        def extract_excel_content_to_document(self):
            try:
                with open(f"./input/{self.file.name}", 'wb') as f:
                    f.write(self.file.read())
                    name = self.file.name
                    loader = UnstructuredExcelLoader(f'./input/{self.file.name}', mode="elements")
                    docs = loader.load()
                    if not docs:
                        raise EmptyFile("No documents found in the Excel file.")
            except FileNotFoundError as e:
                st.error(f"File not found: {e}")
            except Exception as e:
                st.error(f"An error occurred on extract Excel file: {str(e)}")
                return None
            return docs


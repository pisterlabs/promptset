import os
from langchain.document_loaders import CSVLoader
from langchain.document_loaders.pdf import PDFPlumberLoader
from langchain.schema import Document
from pathlib import Path

class MyDirectoryLoader:

    def __init__(self, dir_path):
        if type(dir_path) is str:
            dir_path = Path(dir_path)
            
        self.dir_path = dir_path

    def loadOLd(self):
        docs = []
        for root, _, files in os.walk(self.dir_path):
            for file in files:
                print('file:', file)
                file_path = os.path.join(root, file)
                if file_path.endswith('.csv'):
                    loader = CSVLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PDFPlumberLoader(file_path)
                else:
                    print(f"Do not process the file: {file_path}")
                    continue
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
        return docs
    
    def load(self):
        docs = []
        
        for obj in self.dir_path.rglob('*'):
            if obj.is_file():
                print('file:', obj.name)
                file_path = obj

                if file_path.suffix == '.csv':
                    loader = CSVLoader(str(file_path))
                elif file_path.suffix == '.pdf':
                    loader = PDFPlumberLoader(str(file_path))
                else:
                    print(f"Do not process the file: {file_path}")
                    continue

                loaded_docs: List[Document] = loader.load()
                docs.extend(loaded_docs)

        return docs

    
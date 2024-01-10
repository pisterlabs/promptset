"""
    To give your RAG application access to any user-specific data you first need to load the data. LangChain
    is able to load over 100 different types of documents and data, which is the first step of preparing the
    data for use in your RAG app. Note that some forms of data will require 3rd party packages in order
    for the LangChain loader classes to work.
"""

"""
    The simplest kind of data you can load is text files. LangChain has a TextLoader class that can 
    handle this process without any third party packages. Printing the loaded document to the console will
    show a Document object that has page_content and metadata (just a source in this case).
"""

from langchain.document_loaders import TextLoader
loader = TextLoader("src/resources/lab_docs/simple_text.txt")
doc = loader.load()
print(doc)

"""
    All documents loaded by LangChain will follow this pattern: create your loader, provide the location of the
    file/s and any loading configurations, load the docuemnt/s, and do something with them. here is an example 
    of loading multiple CSV files from a directory using the DirectoryLoader and CSVLoader in tandem
"""

from langchain.document_loaders import CSVLoader, DirectoryLoader
"""
    DirectoryLoader has a convenience argument called "show_progess" that will show a progress bar
    in the console while loading data (set to False by default). If you don't provide an argument for
    loader_cls then an UnstructuredLoader will be used by default, which can lead to unintended behavior.
    
    The DirectoryLoader will load all files in the specified directory and return a list of Documents. All files
    other than hidden files are returned, so if you need to filter out files that you don't want to load you
    have to provide an argument for the "glob" parameter
"""
loader = DirectoryLoader("src/resources/genres_organized", show_progress=True, loader_cls=CSVLoader, glob="**/*.csv")
docs = loader.load()
print(docs, end="\n\n")

"""
    printing the docs produces a mess of csv information, but you can see that it's a collection of rows
    from the CSV data in the "genres_organized" directory in the "resources" directory. With the data now loaded
    you can do whatever you want with it: further organize it, prune it, etc.
    
    The example below takes the documents loaded above and filters them down to only those with the "adventure"
    keyword in the metadata (due to the csv being called "adventure.csv")
"""
adventure_films = [doc for doc in docs if "adventure" in doc.metadata["source"]]
print(adventure_films)

"""
    Implement the methods below to complete the lab
"""

def load_a_text_file(path: str):
    # TODO: implement the function so that a text file can be loaded via the TextLoader class and the document
    #       returned after loading
    pass


def load_a_pdf_file(path: str):
    # TODO: implement the function so that a pdf file can be loaded via the PyPDFLoader class and the document
    #       returned after loading
    pass

def load_csv_files_only_from_directory(path: str):
    # TODO: implement the function so that only csv files are uploaded from a directory and the documents
    #       returned after loading. Use the DirectoryLoader class as your primary loader
    pass
"""
*** PCnewsGPT Wissensimporter - import.py ***
    Änderung: V0.1.4.x - PyMuPDFLoader ersetzt pdfMiner.six
    Änderung: V0.1.3.x - append als funktion, besser lesbare, verständliche chunk-texte
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
from ast import literal_eval
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')
source_directory = os_environ.get('SOURCE_DIRECTORY','source_documents')
append_directory = os_environ.get('APPEND_DIRECTORY','append_documents')
text_splitter_name = os_environ.get('TEXT_SPLITTER','RecursiveCharacterTextSplitter')
text_splitter_parameters = literal_eval(os_environ.get('TEXT_SPLITTER_PARAMETERS','{"chunk_size": 500, "chunk_overlap": 50}'))

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensimporter V0.1.4\n")

"""
Map file extensions to document loaders and their arguments
"""
from langchain.docstore.document import Document as langchain_Document
from langchain.document_loaders import (
    PyMuPDFLoader,
    TextLoader
)
loader_mappings = {
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}

"""
Initialize Text Splitter
"""
# dynamically import the langchain text splitter class and instantiate it
from importlib import import_module
text_splitter_module = import_module("langchain.text_splitter")
TextSplitter = getattr(text_splitter_module, text_splitter_name)
text_splitter = TextSplitter(**text_splitter_parameters)

"""
Initialize Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings
print(f"Embeddings {embeddings_model_name} werden eingelesen...\n")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

"""
Initialize ChromaDB
"""
from langchain.vectorstores import Chroma
from chromadb.config import Settings as Chroma_Settings
# Define the Chroma settings
chroma_settings = Chroma_Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

"""
Checks if ChromaDB exists
"""
from os import system as os_system, path as os_path
from glob import glob
def does_db_exist() -> bool:
    if os_path.exists(os_path.join(persist_directory, 'index')):
        if os_path.exists(os_path.join(persist_directory, 'chroma-collections.parquet')) and os_path.exists(os_path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob(os_path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob(os_path.join(persist_directory, 'index/*.pkl'))
            return True
    return False

"""
parse source_directory (for full import) + append_directory for all filenames to load
"""
full_import_paths = []
for ext in loader_mappings:
    full_import_paths.extend(
        glob(os_path.join(source_directory, f"**/*{ext}"), recursive=True)
    )
append_paths = []
for ext in loader_mappings:
    append_paths.extend(
        glob(os_path.join(append_directory, f"**/*{ext}"), recursive=True)
    )

"""
Load and convert file_path into a langchain document
"""
from regex import sub as regex_sub
def load_file(file_path: str) -> langchain_Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in loader_mappings       # file_paths should alread only be supported types
    # load via langchain
    loader_class, loader_args = loader_mappings[ext]
    loader = loader_class(file_path, **loader_args)
    lc_doc = loader.load()

    # Tidy up PDFs - this might need optimization for some publisher-types
    if ext == ".pdf":
        for i in range(len(lc_doc)):
            # Tidy up PDFs
            doc = lc_doc[i].page_content
            source = lc_doc[i].metadata['source']
            page_num = lc_doc[i].metadata.get('page', None)
            creationdate = lc_doc[i].metadata.get('creationdate', None)

            # some Substutions are independent of PDF-Generators
            # remove line-break hyphenations
            doc = regex_sub(r'-\n+ *', '',doc)
            # remove training spaces in lines
            doc =doc.replace(' \n', '\n')
            # substiture some strange characters
            doc = doc.replace('€', 'Euro')
            doc = doc.replace("„", '"')             # Anführungszeichen-Anfang
            doc = doc.replace("—", '-')             # m-dash
            doc = doc.replace("'", '"')             # replace single with double quotes
            doc = doc.replace("\t", " ")            # replace tabs with a space
            doc = doc.replace("\r", "")             # delete carriage returns
            doc = doc.replace("\v", "")             # delete vertical tabs
            # change single \n in content to " ", but not multiple \n
            doc = regex_sub(r'(?<!\n)\n(?!\n)', ' ',doc)
            # change multiple consecutive \n in content to just one \n
            doc = regex_sub(r'\n{2,}', '\n',doc)
            # remove strange single-characters with optional leading and trailing spaces in lines
            doc = regex_sub(r'\n *(\w|\*) *\n', '\n',doc)
            # remove strange single-characters with spaces inbetween texts
            doc = regex_sub(r'((\w|/|:) +){3,}(\w|/|:)', '',doc)
            # remove multiple blanks
            doc = regex_sub(r'  +', ' ',doc)

            # some Substutions are dependent on "publisher"
            producer = lc_doc[i].metadata.get('producer', '*unknown*')
            if producer.find('Publisher') >= 0:
                # we got a Microsoft Publisher type file
                # substitute known ligatures & strange characters
                doc = doc.replace('(cid:297)', 'fb')
                doc = doc.replace('(cid:322)', 'fj')
                doc = doc.replace('(cid:325)', 'fk')
                doc = doc.replace('(cid:332)', 'ft')
                doc = doc.replace('(cid:414)', 'tf')
                doc = doc.replace('(cid:415)', 'ti')
                doc = doc.replace('(cid:425)', 'tt')
                doc = doc.replace('(cid:426)', 'ttf')
                doc = doc.replace('(cid:427)', 'tti')
                doc = doc.replace('\uf0b7', '*')
                doc = doc.replace('•', '*')
                doc = doc.replace('\uf031\uf02e', '1.')
                doc = doc.replace('\uf032\uf02e', '2.')
                doc = doc.replace('\uf033\uf02e', '3.')
                doc = doc.replace('\uf034\uf02e', '4.')
                doc = doc.replace('\uf035\uf02e', '5.')
                doc = doc.replace('\uf036\uf02e', '6.')
                doc = doc.replace('\uf037\uf02e', '7.')
                doc = doc.replace('\uf038\uf02e', '8.')
                doc = doc.replace('\uf039\uf02e', '9.')
                doc = doc.replace('\uf0d8', '.nicht.')
                doc = doc.replace('\uf0d9', '.und.')
                doc = doc.replace('\uf0da', '.oder.')
                doc = doc.replace('→', '.impliziert. (Mathematisch)')
                doc = doc.replace('\uf0de', '.impliziert.')
                doc = doc.replace('↔', '.äquivalent. (Mathematisch)')
                doc = doc.replace('\uf0db', '.äquivalent.')
                doc = doc.replace('≈','.annähernd.')
                doc = doc.replace('\uf061', 'Alpha')
                doc = doc.replace('β', 'Beta')
                doc = doc.replace('\uf067', 'Gamma')
            elif producer.find('Print To PDF') >= 0:
                doc = doc.replace('�', '_')

            # split doc-content into pages and remove any trailing empty pages
            pages =doc.split('\x0c')
            while (len(pages) > 1) and (pages[-1] == ''):
                pages.pop()
            # if there are no remaining pages, empty the text in lc_doc
            if len(pages) == 0:
                lc_doc[i].page_content = ""
            # if its just one page, update lc_doc with processed content
            elif len(pages) == 1:
                lc_doc[i].page_content = pages[0]
                lc_doc[i].metadata = {"source": f"{source}"}
                if page_num is not None:
                    lc_doc[i].metadata.update({"page": f"{page_num+1}"})
                if creationdate is not None:
                    lc_doc[i].metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
            # if there are muliple pages, create new lc_doc from non-empty pages
            else:
                del lc_doc[i]    # we need a new name + content as multiple pages
                pg_num=0
                for page in pages:
                    if page != '':    # only add non-empty pages, keep page numbering
                        metadata = {"source": f"{source}", "page": f"{pg_num+1}"}
                        if creationdate is not None:
                            metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
                        lc_doc.insert(i, langchain_Document(
                            page_content = page,
                            metadata=metadata
                        ))
                        i+=1
                    pg_num += 1

    # remove empty documents
    docs=[]
    for i in range(len(lc_doc)):
        if lc_doc[i].page_content != '':
            docs.append(lc_doc[i])
    # return the loaded doc               
    return docs

"""
Load + process all documents
"""
# decide if we append to existing db or create a new one
if does_db_exist():
    if len(append_paths) > 0:
        print(f"Dokumentdateien in {append_directory} werden eingelesen und verarbeitet...\n")
        create_db = False
        file_paths = append_paths
        move_from_append = True
    else:
        print(f"Es existiert bereits eine Wissensdatenbank in {persist_directory}.\n")
        print(f"Um Dokumente hinzuzufügen, lege diese im Ordner {append_directory} ab und starte den Import erneut.\n")
        print(f"Um eine neue Wissensdatenbank anzulegen, lösche den Ordner {persist_directory} und starte den Import erneut.\n")
        exit()
else:
    print(f"{persist_directory} wird gelöscht und neu erzeugt.\n")
    os_system(f'rm -rf {persist_directory}')
    print(f"Dokumentdateien in {source_directory} werden eingelesen und verarbeitet...\n")
    create_db = True
    file_paths = full_import_paths
    move_from_append = False


# process all documents
db = None
total_chunks=0
for idx,file_path in enumerate(file_paths):
    print(f"Datei {file_path} ({idx+1}/{len(file_paths)})...")
    documents=load_file(file_path)                  # txt as 1 document, pdfs as 1 document per page
    print(f"... wurde eingelesen und in {len(documents)} Seite(n) umgewandelt ...")
    # Split into chunks of text
    chunks = text_splitter.split_documents(documents)
    print(f"... zerteilt auf {len(chunks)} Textteil(e) ...")
    total_chunks += len(chunks)
    # tidying-up chunk text and numbering of chunks per source
    n_chunk=1
    chunk_metadata=chunks[0].metadata
    for i in range(len(chunks)):
        # new chunk begins if metadata is different
        if chunk_metadata!=chunks[i].metadata:
            n_chunk=1
            chunk_metadata=chunks[i].metadata
        else:
            n_chunk += 1
        chunks[i].metadata.update({"chunk": f"{n_chunk}"})

        # final tidying-up of chunk text - needed because of SpaCy weirdnesses
        txt = chunks[i].page_content
        # change single \n in content to " ", but not multiple \n
        txt = regex_sub(r'(?<!\n)\n(?!\n)', ' ',txt)
        # change multiple consecutive \n in content to just one \n
        txt = regex_sub(r'\n{2,}', '\n',txt)
        # replace txt if changed
        if txt != chunks[i].page_content:
            chunks[i].page_content = txt
               
    # create embeddings and persist
    if db is None:
        # create or append-to db
        if create_db:
            db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory, client_settings=chroma_settings)
        else:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=chroma_settings)
            db.add_documents(chunks)
    else:
        # add to existing db
        db.add_documents(chunks)
    db.persist()
    print("... und in der Wissensdatenbank gespeichert.\n")

# move files from append_directory to source_directory after finishing import
if move_from_append:
    print(f"Verschiebe alle Dateien aus {append_directory} nach {source_directory}...\n")
    os_system(f'mv {append_directory}/* {source_directory}/')

# Statistics
print(f"Insgesamt {len(file_paths)} Dokument(e) mit {total_chunks} Textteil(en) wurden eingelesen.")
db = None

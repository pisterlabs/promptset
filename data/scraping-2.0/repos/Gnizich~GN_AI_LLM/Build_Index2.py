import os
import hashlib
from threading import Thread
from pathlib import Path
#import llama_index
from openai import OpenAI
import constants as c
c.Get_API()
client = OpenAI()

newdocspath = ""
masterpath = ""
basepath = ""
persistpath = ""
# test
class Document:
    __slots__ = ['text', 'doc_id', 'id_', 'hash']

    def __init__(self, text: str, doc_id: str):
        self.text = text
        self.doc_id = doc_id
        self.id_ = doc_id
        self.hash = self.generate_hash(text)

    def generate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_metadata_str(self, mode=None) -> str:
        return f"{self.doc_id}-{self.hash}"

    def get_content(self, metadata_mode=None) -> str:
        return self.text


def index_document(doc: Document):
    print("index_document reached")
    index = VectorStoreIndex()
    index.add_document(doc)
    print("index doscument complete")

def CreateUpdate_Index(basepath, masterdocs, newdocs, indexpath, action, tool ):
    print('Create/Update function running')

    # Ask questions until user exits
    while True:
        # Check if index path directory is empty
        chkindexpath = "Z:\\MyChatBot_v1.0\\"+ tool + "\\index\\"
        print(chkindexpath)
        index_dir = Path(chkindexpath)
        is_empty = len(os.listdir(index_dir)) == 0

        if is_empty:
            print('Running creating index function')
            Create_Index(basepath, masterdocs, newdocs, indexpath, tool )
        else:
            print('Running updating index function')
            Update_Index(basepath, masterdocs, newdocs, indexpath)


def Create_Index(basepath: str, masterdocs: str, newdocs: str, indexpath: str, tool):
    print('Creating index')
    from llama_index import StorageContext, VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, Document

    # Specify the input_dir path
    docpath = masterdocs
    documents = SimpleDirectoryReader(input_dir=docpath).load_data()

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents)

    # Persist index to disk
    saveindexpath = basepath + indexpath
    index.storage_context.persist(saveindexpath)

    print('Index created and saved')

    docs_dir = os.path.join("Z:\\MyAI_Training_Docs\\", tool, "_Training_Docs\\docs")
    doc_paths = Path(docs_dir).glob("*")

    num_nodes = 8
    nodes = [BaseNode() for _ in range(num_nodes)]
    index = VectorStoreIndex(nodes=nodes)

    threads = []
    for path in doc_paths:
        with open(path) as f:
            text = f.read()
        doc = Document(text, path.name)
        thread = Thread(target=index_document, args=(doc,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    storage_context = StorageContext(indexdir=indexpath)
    storage_context.persist(index)
    print("Create index complete")

def Update_Index(basepath: str, masterdocs: str, newdocs: str, indexpath: str):
    print("update index reached")
    import os
    from llama_index import load_index_from_storage

    storage_context = StorageContext.from_defaults(indexpath)
    index = load_index_from_storage(storage_context)

    new_docs_dir = os.path.join(basepath, newdocs)
    for filename in os.listdir(new_docs_dir):
        path = os.path.join(new_docs_dir, filename)
        with open(path) as f:
            text = f.read()
        doc = Document(text, filename)
        index.add_document(doc)

    storage_context.persist(index)
    print("Update index completed")

def AskBuild(tool, choice):
    print("AskBuild reached : ", tool, choice)
    if choice == 'build':
        print("Askbuild build reached")
        basepath = 'Z:\\MyAI_Training_Docs\\'
        persistpath = 'Index\\Index\\'
        if tool == 'ai':
            doc_path = "AI"
        elif tool == 'gn':
            doc_path = "GN"
        newdocspath = basepath + doc_path + "_Training_Docs\\Docs"
        masterpath = basepath + doc_path + "_Training_Docs\\Master"
        print(tool, choice)
        print("PP: ", persistpath)
        print("nd: ", newdocspath)
        print("mp: ", masterpath)
        print("bp: ", basepath)
        CreateUpdate_Index(basepath, masterpath, newdocspath, persistpath, choice, tool)
        print("Askbuild GN complete")
    elif choice == 'ask':
        print("Askbuild ask reached")
        persistpath = 'Index\\Index\\'
        newdocspath = 'Docs'
        masterpath = 'Master'
        basepath = 'Z:\\MyChatBot_v1.0\\' + tool + '\\'
        AskQuestion(basepath, persistpath)
        print("Ask build ask complete")
    else:
        pass


def AskQuestion(indexpath: str):
    print("Ask question reached")
    storage_context = StorageContext.from_defaults(indexpath)
    index = load_index_from_storage(storage_context)

    while True:
        question = input("Enter question: ")
        if question.lower() == "exit":
            break

        response = index.query(question)
        print(response)
    print("AskQuestion complete")
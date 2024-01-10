import os
import hashlib
from threading import Thread
from pathlib import Path
#import llama_index
from openai import OpenAI
import constants as c
from llama_index import StorageContext, VectorStoreIndex, Document
from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader

c.Get_API()
client = OpenAI()

newdocspath = ""
masterpath = ""
basepath = ""
persistpath = ""
indexpath = ""
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

    # Check if index path directory is empty
    main_dir = "."
    indexes_dir = os.path.join(main_dir, "Indexes")
    chkindexpath = os.path.join(indexes_dir, tool)
    print('ckindexpath', chkindexpath)
    index_dir = Path(chkindexpath)
    print('index_dir',index_dir)
    is_empty =len(os.listdir(index_dir)) == 0
    print('is empty', is_empty)
    if is_empty:
        print('Running creating index function')
        print(basepath, masterdocs, newdocs, index_dir, tool)
        Create_Index(basepath, masterdocs, newdocs, index_dir, tool )
    else:
        print('Running updating index function')
        Update_Index(basepath, masterdocs, newdocs, index_dir)
    # print('Running creating index function')
    # print(basepath, masterdocs, newdocs, index_dir, tool)
    # Create_Index(basepath, masterdocs, newdocs, index_dir, tool )

def Create_Index(basepath: str, masterdocs: str, newdocs: str, indexpath: str, tool: str):

    print('Creating index')

    # Load documents
    docpath = masterdocs
    documents = SimpleDirectoryReader(input_dir=docpath).load_data()

    # Parse documents into nodes
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)

    # Create index using nodes
    index = VectorStoreIndex(nodes=nodes)
    for doc in documents:
        index.insert(doc)

    # Persist index
    persist_path = os.path.join(basepath, indexpath)
    print('persist_path= ', persist_path)
    saveindexpath = persist_path
    index.storage_context.persist(saveindexpath)

    print('Index created and saved')

# def Update_Index(basepath: str, masterdocs: str, newdocs: str, indexpath: str):
#     print("update index reached")
#     from llama_index import load_index_from_storage, Document
#     print('update_index indexpath', indexpath)
#
#     try:
#         storage_context = StorageContext.from_defaults(persist_dir=indexpath)
#         new_index = load_index_from_storage(storage_context)
#         new_docs_dir = os.path.join(basepath, newdocs)
#         is_empty = len(os.listdir(newdocs)) == 0
#         if not is_empty:
#             for filename in os.listdir(new_docs_dir):
#                 path = os.path.join(new_docs_dir, filename)
#                 with open(path) as f:
#                     # Create document
#                     text = f.read()
#                     doc = Document(text, filename)
#                     new_index.insert(doc)
#                 storage_context.persist(new_index)
#         print("Update index completed")
#     except Exception as e:
#         print(e)

def Update_Index(basepath: str, masterdocs: str, newdocs: str, indexpath: str):
    # Loading from disk
    from llama_index import StorageContext, load_index_from_storage
    from llama_index import PromptHelper, LLMPredictor, ServiceContext
    import openai
    openai.api_key = c.Get_API()

    is_empty =len(os.listdir(newdocs)) == 0

    if not is_empty:
        storage_context = StorageContext.from_defaults(persist_dir=indexpath)
        index = load_index_from_storage(storage_context)
        new_docs_dir = os.path.join(basepath, newdocs)
        llm_predictor =LLMPredictor(llm=openai)
        max_input_size = 4096
        num_outputs = 5000
        max_chunk_overlap = 0.5
        chunk_size_limit = 3900
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        reader = SimpleDirectoryReader(new_docs_dir)
        documents = reader.load_data()
        persist_path = persist_path = os.path.join(basepath, indexpath)
        for d in documents:
            index.insert(document = d, service_context = service_context)
        print(persist_path)
        storage_context.persist(persist_dir = persist_path)
    else:
        print('no new docs')



def AskBuild(tool, choice):
    print("AskBuild reached : ", tool, choice)
    if choice == 'build':
        print("Askbuild build reached")
        main_dir = "."
        #train_dir = os.path.join(main_dir, "MyAI_Training_Docs")
        train_dir = ".//MyAI_Training_Docs//"
        train_path = os.path.join(train_dir, tool)
        master_dir = os.path.join(train_path, "Master")
        persistpath = 'Indexes//' + tool + '//'
        if tool == 'ai':
            doc_path = "ai"
        elif tool == 'gn':
            doc_path = "gn"
        newdocspath = train_path + "//Docs"
        masterpath = train_path + "//Master"
        print(tool, choice)
        print("PP: ", persistpath)
        print("nd: ", newdocspath)
        print("mp: ", masterpath)
        #print("bp: ", basepath)
        basepath = ""
        CreateUpdate_Index(basepath, masterpath, newdocspath, persistpath, choice, tool)
        print("Askbuild gn complete")
    elif choice == 'ask':
        print("Askbuild ask reached")
        persistpath = 'Indexes//'
        newdocspath = 'Docs'
        masterpath = 'Master'
        main_dir = "."
        basepath = os.path.join(main_dir, tool)
        indexpath = main_dir + '//Indexes//' + tool + '//'
        AskQuestion(indexpath, persistpath)
        print("Ask build ask complete")
    else:
        pass


def AskQuestion(topic: str, action: str, question: str):
    from llama_index import load_index_from_storage
    print(topic)
    print("Ask question reached")
    indexpath = './/Indexes//' + topic + '//'
    print('indexpath= ', indexpath)
    print(os.listdir(indexpath))
    storage_context = StorageContext.from_defaults(persist_dir=indexpath)
    new_index = load_index_from_storage(storage_context)
    new_query_engine = new_index.as_query_engine()

    while True:
        if question.lower() == "exit":
            break
        response = new_query_engine.query(question)
        print(response)

        print("AskQuestion complete")
        return response

#AskBuild('gn', 'build')

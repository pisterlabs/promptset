import os.path

import chromadb
import langchain.embeddings
import win32com.client
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import json

from os.path import isdir, isfile, join
from os import listdir
import openai
from optiondata import Option_data
from signalManager import SignalManager

loaded = False
dbFolder = './vectorstore/'
workspace = './workspace/'
metadatas = dict()
chroma: Chroma

option_data = Option_data()
openai.api_key = option_data.openai_api_key
option_data.optionSignals.changed_checked_api.connect(lambda: reloadDB())


def getExtension(fname: str) -> str:
    spl = fname.split('.')
    if len(spl) == 1:
        return ''

    return spl[-1]


def processFile(rootpath, path, fname, documents):
    ext = getExtension(fname)
    allPath = join(rootpath, path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        separators=['\n\n', '\n', ' ', '']
    )

    if ext == 'txt':
        loader = TextLoader(join(allPath, fname), encoding='utf8')
        document = loader.load()
        print('loaded file {0} with TextLoader.'.format(fname))
    elif ext == 'pdf':
        loader = UnstructuredPDFLoader(join(allPath, fname))
        document = loader.load()
        print('loaded file {0} with UnstructuredPDFLoader.'.format(fname))
    else:
        print("can't process file {0}".format(fname))
        return
    documents.extend(text_splitter.split_documents(document))


def CheckFile(rootpath, path, fname):
    metaResult = checkMetadata(rootpath, path, fname)
    if metaResult == 2:  # New File
        print('file: {0} has added to workspace'.format(fname))
        createNewDocument(rootpath, path, fname)
    elif metaResult == 1:  # File modified
        print('file: {0} needs to be updated'.format(fname))
        updateDocument(rootpath, path, fname)
    else:  # Recent file
        print('file: {0} is up to date'.format(fname))


def iterateDirectory(rootpath, path):
    allPath = join(rootpath, path)
    dirs = [d for d in listdir(allPath) if isdir(join(allPath, d))]
    files = [f for f in listdir(allPath) if isfile(join(allPath, f))]
    for d in dirs:
        iterateDirectory(rootpath, join(path, d))

    for f in files:
        CheckFile(rootpath, path, f)


def get_file_metadata(path, filename):
    return os.path.getmtime(join(os.path.abspath(path), filename))


def create_or_update_metadata(workspacePath, filePath, fileName, docID, idxNum):
    global metadatas
    path = join(workspacePath, filePath)
    meta = {'path': join(filePath, fileName),
            'modified': get_file_metadata(path, fileName),
            'docID': docID,
            'idxNum': idxNum}
    metadatas['files'][join(filePath, fileName)] = meta


def checkMetadata(workspacePath, filePath, fileName) -> int:  # 0 : same, 1 : not same, 2 : not found
    global metadatas
    file = join(filePath, fileName)
    if file not in metadatas['files']:
        return 2

    modified_origin = metadatas['files'][file]['modified']
    path = join(workspacePath, filePath)
    modified = get_file_metadata(path, fileName)

    if modified == modified_origin:
        return 0
    else:
        return 1


def createNewDocument(workspacePath, filePath, fileName):
    global chroma
    global metadatas
    idx = metadatas['lastID']
    metadatas['lastID'] = metadatas['lastID'] + 1
    docs = []
    processFile(workspacePath, filePath, fileName, docs)
    ids = []
    for i in range(len(docs)):
        ids.append('{0}d{1}'.format(idx, i))
    embedding = OpenAIEmbeddings()
    if len(docs) != 0:
        chroma.add_documents(documents=docs, ids=ids)
        create_or_update_metadata(workspacePath, filePath, fileName, idx, len(docs))


def updateDocument(workspacePath, filePath, fileName):
    global chroma
    global metadatas
    file = join(filePath, fileName)
    docs = []
    processFile(workspacePath, filePath, fileName, docs)
    idx = metadatas['files'][file]['docID']
    idNum = metadatas['files'][file]['idxNum']
    coll = chroma._client.get_collection('langchain')
    ids = []
    newIds = []
    for i in range(idNum):
        ids.append('{0}d{1}'.format(idx, i))
    for i in range(len(docs)):
        newIds.append('{0}d{1}'.format(idx, i))
    coll.delete(ids=ids)
    embedding = OpenAIEmbeddings()
    chroma.add_documents(documents=docs, ids=newIds)
    create_or_update_metadata(workspacePath, filePath, fileName, idx, len(docs))


def initMetadata():
    global metadatas
    metadatas['files'] = dict()
    metadatas['lastID'] = 0
    saveMetadata(dbFolder + '/metadata.json')


def saveMetadata(path):
    global metadatas
    with open(path, "w") as f:
        json.dump(metadatas, f,
                  indent=4)


def loadMetadata(path):
    global metadatas
    with open(path, "r") as f:
        metadatas = json.load(f)
        f.close()


def createDB():
    global chroma
    from chromadb.config import Settings
    embedding = OpenAIEmbeddings()
    chroma = Chroma(
        persist_directory=dbFolder,
        embedding_function=embedding
    )
    initMetadata()
    iterateDirectory(workspace, '')
    saveMetadata(dbFolder + 'metadata.json')
    chroma.persist()


def loadDB():
    global loaded
    if loaded:
        return
    global chroma

    try:
        embedding = OpenAIEmbeddings()
        chroma = Chroma(persist_directory=dbFolder, embedding_function=embedding)
        loadMetadata(join(dbFolder, 'metadata.json'))
        iterateDirectory(workspace, '')
        saveMetadata(join(dbFolder, 'metadata.json'))
        chroma.persist()
        loaded = True
    except:
        print('failed to loadDB')


def reloadDB():
    print('reloading DB')
    option_data.load_option()
    openai.api_key = option_data.openai_api_key
    loadDB()

def promptLangchain(query):
    global chroma
    if chroma is None:
        print("chroma didn't set")
        return 'err'
    retriever = chroma.as_retriever()
    openai = OpenAI()
    openai.max_tokens = 256
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=option_data.temperature),
        chain_type='stuff',
        retriever=retriever
    )

    return qa.run(query)

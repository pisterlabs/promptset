from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import CHROMA_CLIENT, EMBEDDING_FUNC, LAW_COLLECTION_NAME
from chromadb.api.models.Collection import Collection
import os, shutil, docx, re, time
from werkzeug.datastructures import FileStorage
from io import BytesIO
from ocr import load_pdf
import unicodedata

def upsert_text(collection_name:str, text:str, filename:str, case_name, chunk_size=1000, chunk_overlap=100):
    # from langchain.text_splitter import NLTKTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['.', '\n\n', '\n', ',', '。','，'])
    # text_splitter = NLTKTextSplitter()
    try:
        collection = CHROMA_CLIENT.get_or_create_collection(collection_name)
        chunks = text_splitter.split_text(text)
        print("chunks:", len(chunks))
        pattern = re.compile(r'[\n\r\t]')   # necessary to process chinese 换行符
        time_stamp = time.time()
        for i,t in enumerate(chunks, start=1):
            txt = unicodedata.normalize('NFKC', re.sub(pattern, ' ', t))    # remove full-width space and satro
            collection.upsert(
                embeddings = [EMBEDDING_FUNC.embed_query(txt)],  # if using OpenAIEmbedding, do not need [0]
                documents = [txt],
                metadatas = [{"source": filename, "doc_type": case_name, "valid": 1, "timestamp": time_stamp}],
                ids = [filename+'-'+str(i)]
            )
    except Exception as e:
        print("upload error on ", type(e))
        print(e.args)
        print(e)
        raise SystemExit(1)
    return "success"

# Upload docs in a dir to vectorDB and move the file to a different folder
def init_law_db(category: str, dir:str):
    from os import walk
    if not os.path.exists(dir + "loaded"): os.mkdir(dir + "loaded")

    # load all files in a folder
    for fn in next(walk(dir), (None, None, []))[2]:  # [] if no file
        text = ""
        f_name, f_ext = os.path.splitext(fn)
        if f_ext.lower()=='.pdf':
            print("Reading:", fn)
            fo = open(dir+fn, "rb")
            text += load_pdf(fo.read())
        elif f_ext.lower()=='.docx':
            print("Reading:", fn)
            for line in docx.Document(dir+fn).paragraphs:
                text += "\n"+line.text
        elif f_ext.lower()=='.txt':
            print("Reading:", fn)
            fo = open(dir+fn)
            for line in fo.readlines():
                text += line
        else:
            continue

        print(text[:100])
        upsert_text(LAW_COLLECTION_NAME, text, f_name, category)

        # move the file to other folder once it is done
        if fo: fo.close()
        shutil.move(dir+fn, dir+"loaded")

# Process file upload from socket client. Referred by flask service code
def extract_text(filename, filetype, filedata):
    file = FileStorage(
        stream=BytesIO(filedata), 
        filename=filename,
        content_type=filetype, 
        content_length=len(filedata)
    )
    file_ext = os.path.splitext(filename)[1]
    text = ""
    if file_ext.lower()==".pdf":
        text += load_pdf(file.read())
    elif file_ext.lower()==".docx":
        for line in docx.Document(file).paragraphs:
            text += "\n"+line.text
        # print("text=", text)
    elif file_ext.lower()==".txt":
        for line in file.read().decode('utf8'):
            # print(line)
            text += line
    print("text=", text[:100])
    return text

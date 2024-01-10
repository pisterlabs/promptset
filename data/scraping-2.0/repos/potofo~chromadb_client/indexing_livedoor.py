# chromadb
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
# langchain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# time
import time
# sys 
import sys
# pprint
import pprint
# url detect
import re
# tqdm
import tqdm
from tqdm import tqdm
import numpy as np


#directory = './livedoorニュースコーパス/ldcc-20140209/text/small/'
#directory = './livedoorニュースコーパス/ldcc-20140209/text/movie-enter/'
directory = './livedoorニュースコーパス/ldcc-20140209/text/it-life-hack/'

#loader = DirectoryLoader(directory, glob="movie-enter*.txt",show_progress=True, encoding='utf8')
#loader = TextLoader(directory+"movie-enter-5840081.txt", encoding='utf8')
#loader = DirectoryLoader(directory, glob="movie-enter*.txt",show_progress=True, loader_cls=TextLoader, encoding='utf8')
#loader = DirectoryLoader(directory, glob="movie-enter*.txt", loader_cls=TextLoader, encoding='utf8')
# reffer to https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
# C.Auto detect encodings
text_loader_kwargs={'autodetect_encoding': True}
#loader = DirectoryLoader(directory, glob="movie-enter*.txt", show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
loader = DirectoryLoader(directory, glob="it-life-hack*.txt", show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

start = time.time()
documents = loader.load()
end = time.time()
time_diff = end - start

num_documents = len(documents)
spendding_time = round(time_diff,2)

print('documents:{0} documents spendding {1} seconds.'.format(num_documents,str(round(time_diff,2))))

# scheme of documents
# document[0]
# document[0].page_content
# document[0].metadata
# ...
# document[10]
# document[10].page_content
# document[10].metadata

# for document in documents:
#     print("path:{0}".format(document.metadata['source']))

# pprint.pprint(documents[0: 2],indent=2,width=40)
# pprint.pprint(documents[0].page_content,indent=2,width=40)
# pprint.pprint(documents[0].metadata,indent=2,width=40)

# Number of documents processed at once
num_proc_documents = 20
cur_document_num = 0

bar = tqdm(total = num_documents)
bar.set_description('Progress rate')

start = time.time()

for i in range(0,num_documents,num_proc_documents):
    # tqdm.write("i:{0}".format(i))
    # tqdm.write("i+num_proc_documents:{0}".format(i+(num_proc_documents)))
    # tqdm.write("num_documents:{0}".format(num_documents))
#    if(num_documents % num_proc_documents):
#    num_documents

    proc_documents = documents[cur_document_num:(cur_document_num+num_proc_documents):1]
    # tqdm.write("proc_documents[{0}:{1}:1]".format(cur_document_num,cur_document_num+num_proc_documents))
    # tqdm.write("num proc_documents:{0}".format(len(proc_documents)))
    #pprint.pprint(proc_documents,indent=2,width=40)
    # for doc in proc_documents:
    #     tqdm.write("path:{0}".format(doc.metadata['source']))
    #pprint.pprint(proc_documents,indent=2,width=40)

    try:
        #client = chromadb.HttpClient(
        #    host='localhost',
        #    port=80)

        # With authentifizations
        client = chromadb.HttpClient(
            host='localhost',
            port=80,
            settings=Settings(chroma_client_auth_provider='chromadb.auth.token.TokenAuthClientProvider',
                            chroma_client_auth_credentials='test-token'))
    except Exception as e:
        print('Vector database Connection error occurs with following message.')
        print('Error Message:{0}'.format(str(e)))
        sys.exit(-1)

    # defined sentence transformer LLM
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")

    # get or create collection
    collection = client.get_or_create_collection("livedoor",embedding_function=sentence_transformer_ef)

    items = collection.get()
    #pprint.pprint(items,indent=2,width=40)

    if(len(items['ids']) == 0):
        last_ids = 0
    else:
        last_ids = int(items['ids'][-1])

    #print('last ids:{0}'.format(last_ids))


    # split chunk each num_proc_documents
    chunk_size=512
    chunk_overlap=20
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(proc_documents)

    tqdm.write("{0} chunks in {1} documents".format(len(chunks),len(proc_documents)))

    # scheme of Chunk
    # chunks[0]
    # chunks[0].page_content
    #   "http://news.livedoor.com/article/detail/5840081/\n\n2011\n\n09\n\n08T10:00:00+0900\n\nインタビュー：宮崎あおい..."
    # chunks[0].metadata
    #   {'source':'livedoorニュースコーパス\\ldcc-20140209\\text\\small\\movie-enter-5840081.txt'}
    # ...
    # chunks[10]
    # chunks[10].page_content
    # chunks[10].metadata

    # Initialization Scheme of Vector Database
    vect_documents = []
    vect_metadatas = []
    vect_ids       = []

    # Restructure chunks
    cur_chunk_num = 0
    # defined current ids number
    cur_ids = last_ids + 1
    for chunk in chunks:
    #   strip_docs.append(chunk.page_content)
        splitline_chunk_page_content = chunk.page_content.splitlines()
        check_url  = re.findall('^https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', splitline_chunk_page_content[0])

        if(check_url):
            # first chunk of document
            # print("found url:{0}".format(check_url[0]))
            # tqdm.write("document_url    :{0}".format(check_url[0]))
            document_url = check_url[0]
            document_date  = splitline_chunk_page_content[1]
            document_title = splitline_chunk_page_content[2]
            target_page_content = splitline_chunk_page_content[3]
            #cur_document = chunk.page_content[index_document+1]
            #cur_document = chunk.page_content
            index_document = chunk.page_content.find(target_page_content)
            #print(f'target_page_content:{target_page_content}')
            #print(f'index_document     :{index_document}')
            cur_document = chunk.page_content[index_document:]
            cur_chunk_num = 0
            #bar.update(cur_document_num/num_documents)
            
            cur_document_num+=1
            # tqdm.write("cur_document_num:{0}".format(cur_document_num))
            bar.update(1)
            

        else:
            # Second and subsequent chunks
            cur_document = chunk.page_content

        # print(f'cur_chunk_num:{cur_chunk_num}')
        # tqdm.write(f'cur_chunk_num:{cur_chunk_num}')
        # print(f'cur_document:{cur_document}')
        cur_document = cur_document.replace('\u200b', '')
        cur_document = cur_document.replace('\u3000', '')
        vect_documents.append(cur_document)
        dict_metadatas = {}
        dict_metadatas["url"]   = document_url
        dict_metadatas["date"]  = document_date
        document_title = document_title.replace('\u200b', '')
        document_title = document_title.replace('\u3000', '')
        dict_metadatas["title"] = document_title
        dict_metadatas["chunk"] = cur_chunk_num
        vect_metadatas.append(dict_metadatas)
        vect_ids.append(str(cur_ids))
        cur_ids+=1

        collection.add(
            ids=vect_ids,
            metadatas=vect_metadatas,
            documents=vect_documents
        )
                
        cur_chunk_num = cur_chunk_num + 1
        #bar.update(cur_document_num/num_documents)
        #bar.update(cur_document_num)

end = time.time()
time_diff = end - start
print('documents:{0} documents spendding {1} seconds.'.format(num_documents,str(round(time_diff,2))))


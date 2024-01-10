import os, types
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import pinecone
from chromadb.config import Settings

OPENAIKEY= os.environ.get('OPENAIKEY')
openai.api_key = OPENAIKEY
PINECONEKEY = os.environ.get('PINECONEKEY')


chroma_collection_st = "Books_Index001"
chroma_collection_oai = "Books_Index002"
pinecone_index_oai = "index000003"


chroma_client = chromadb.PersistentClient(path="/tmp/cdb")
collection_st = chroma_client.get_or_create_collection(chroma_collection_st)
collection_oai = chroma_client.get_or_create_collection(chroma_collection_oai)
pinecone.init(api_key=PINECONEKEY,environment='northamerica-northeast1-gcp') 
pcindex = pinecone.Index(pinecone_index_oai)

stembeddingsmodel = SentenceTransformer('all-MiniLM-L6-v2')

def prepare_embedding_function(data):

    language = str(data[6])
    pages = str(data[12])
    publishDate = str(data[14])
    firstPublishDate = str(data[15])    
    price = str(data[24])
    series = str(data[2])
    bookFormat = str(data[10])
    publisher = str(data[13])
    edition = str(data[11])
    coverImg = str(data[21])
    description = str(data[5])
    
    if language == "nan":
        language = ""    
    if firstPublishDate == "nan":
        publishDate = ""
    if firstPublishDate == "nan":
        firstPublishDate = ""
    if pages == "nan":
        pages = ""               
    if price == "nan":
        price = 0.0
    if series == "nan":
        series = ""
    if bookFormat == "nan":
        bookFormat = ""        
    if publisher == "nan":
        publisher = ""    
    if edition == "nan":
        edition = ""    
    if description == "nan":
        description = ""                 
        
    edict = {
        "title":data[1],
        "series":series,
        "author":data[3],
        "rating":data[4],
        "description":description,
        "language":language,
        "isbn":data[7],
        "genres":data[8],
        "characters":data[9],
        "bookFormat":bookFormat,
        "edition":edition,
        "pages":pages,
        "publisher":publisher,
        "publishDate":publishDate,
        "firstPublishDate":firstPublishDate,
        "awards":data[16],
        "numRatings":data[17],        
        "ratingsByStars":data[18],
        "likedPercent":data[19],
        "setting":data[20],
        "coverImg":coverImg,
        "bbeScore":data[22],
        "bbeVotes":data[23],
        "price":price
        
        }
    return edict    
# Embeddings Choice 1 : Sentence Transformer
def embedding_function(data):
   embeddings = stembeddingsmodel.encode(data)
   return embeddings.tolist()

# Embeddings Choice 2 : OAI text-embedding-ada-002
def oai_embedding_function(data):
    #print(embeddingsData)
    response = openai.Embedding.create(
    input=data,
    model="text-embedding-ada-002")
    #print(response)
    embeddings = list(map(lambda row: row.embedding,response.data))
    return embeddings

# Begin 
df = pd.read_csv('dataset/csv/ds_books_1k.csv')
datalength = len(df)
batchsize = 100
begin = 0
end = batchsize
t_= 0
while True:
    print(f"{begin} : {end}")
    if begin >= datalength:
        break
    batch = df.iloc[begin:end] 
    #print(batch.count)
    metadata = batch.apply(lambda row: prepare_embedding_function(row), axis=1)   
    embeddable = batch.apply(lambda row: str(prepare_embedding_function(row)), axis=1)   
    
    #Get Embeddings from both models
    embeddings_st = embedding_function(embeddable.to_list())
    #embeddings_oai = oai_embedding_function(embeddable.to_list())
    ids_ = batch['bookId'].apply(lambda row: str(row)).tolist()

    collection_st.upsert(
        embeddings = embeddings_st,
        metadatas = metadata.tolist(),
        ids = ids_
    )
    
    #collection_oai.upsert(
    #    embeddings = embeddings_oai,
    #    metadatas = metadata.tolist(),
    #    ids = ids_
    #)   
    
    #data=list(zip(ids_, embeddings_oai,metadata.tolist()))
    #pcindex.upsert(data)

    t_= t_+ len(batch)
    p_ = str((round(t_/datalength*100,2)))
    print(f"{t_}/{datalength}. {p_}%")
    
    begin = end
    end = end + batchsize
    
print("End OF Embeddings Creation ")  



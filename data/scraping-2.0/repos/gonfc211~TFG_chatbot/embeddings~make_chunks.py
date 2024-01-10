import openai
import pandas as pd
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()  

openai.api_key = os.getenv("OPENAI_API_KEY")

texts = []

captions_directory = os.listdir('captions')

file_names = [
    'captions/'+i for i in captions_directory if i.endswith('.txt')
]



for file_name in file_names:
    with open(file_name, 'r') as file:
        texts.append(file.read())


with open('captions/all_captions.txt', 'w')as outfile:
    for text in texts:
        outfile.write(text)
        outfile.write('\n')
   


r_splitter = r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=100, 
    separators=["\n\n", "\n", " ", ""]
)

with open('captions/all_captions.txt', 'r') as file:
    all_captions = file.read() 


chunks = r_splitter.split_text(all_captions)

#make embeddings
# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(chunks), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = chunks[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": chunks, "embedding": embeddings})


# save document chunks and embeddings

df.to_csv('video_embedd.csv', index=False)

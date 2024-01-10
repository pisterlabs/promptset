import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import AstraDB
import os
from langchain.embeddings import OpenAIEmbeddings



token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
openai_api_key=os.environ["OPENAI_API_KEY"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="movies_description",
    api_endpoint=api_endpoint,
    token=token,
)


filename = 'movies_new.csv'
df = pd.read_csv(filename)

start = 0
end = 200
batch_size = 100
for y in range(start, end, batch_size):
    print(f"Processing {y} to {end} texts")
    batch_start = y
    for i in range(batch_start, batch_start+batch_size, batch_size):
        print(f"Processing {i} to {i+batch_size} llm texts")
        batch = df[i:i+batch_size]
        batch = batch.fillna('')
        texts = []
        metadatas = []
        ids = []
        for id, row in batch.iterrows():
            rawtext = f"Title : {row['title']} Year: {row['year']} Genre: {row['genre']} Description: {row['description']}"
            print(rawtext)
            texts.append(rawtext)
            metadatas.append(row.to_dict())
            ids.append(row['title'])
        inserted_ids = vstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"\nInserted {len(inserted_ids)} documents.")



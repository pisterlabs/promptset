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
    collection_name="ecommerce_inventory",
    api_endpoint=api_endpoint,
    token=token,
)

#url = "https://huggingface.co/datasets/codeparrot/codecomplex/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
#df = pd.read_parquet("train-00000-of-00001-1f042f20fd269c32.parquet")

filename = 'ecommerce_meta_data_final.csv'
df = pd.read_csv(filename)

start = 100000
end = 2140000
batch_size = 100
for y in range(start, end, batch_size):
    print(f"Processing {y} to {end} texts")
    batch_start = y
    #llmtexts = []
    #docs = []
    for i in range(batch_start, batch_start+batch_size, batch_size):
        print(f"Processing {i} to {i+batch_size} llm texts")
        batch = df[i:i+batch_size]
        batch = batch.fillna('')
        texts = []
        metadatas = []
        ids = []
        for id, row in batch.iterrows():
            rawtext = f"Gender : {row['gender']} Season: {row['season']} Price: {row['price']} Title: {row['title']} Image description: {row['blip_large_caption']} Usage: {row['usage']} Year: {row['year']}"
            print(row['title'])
            texts.append(rawtext)
            metadatas.append(row.to_dict())
            ids.append(row['basename'])

            #doc = Document(page_content=rawtext, metadata=row.to_dict())
            #docs.append(doc)
            #print(llmtexts)
       #batch['llmtext'] = llmtexts
        inserted_ids = vstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"\nInserted {len(inserted_ids)} documents.")

    #batch.to_csv(f"llm_product_{start}.csv", encoding='utf-8', index=False)



#print(df.head(5)) 

#base_url = "https://huggingface.co/datasets/rajuptvs/ecommerce_products_clip/blob/main/data/train-00000-of-00001-1f042f20fd269c32.parquet"
#data_files = {"train": base_url}
#ecommercedata = load_dataset("parquet", data_files=data_files, split="train")


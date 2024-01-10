from langchain.docstore.document import Document
from langchain.utilities import ApifyWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ApifyDatasetLoader
import os
import csv
import pandas as pd
from langchain.document_loaders import DataFrameLoader

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
openai_api_key=os.environ["OPENAI_API_KEY"]
#apify_api_key=os.environ["APIFY_API_TOKEN"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="ptc_new_inventory",
    api_endpoint=api_endpoint,
    token=token,
)

filename = 'Product_Website_Load.csv'
df = pd.read_csv(filename)
llmtexts = []
start = 2000
batch_size = 1000
docs = []
for i in range(start, start+batch_size, batch_size):
    print(f"Processing {i} to {i+batch_size} llm texts")
    batch = df[i:i+batch_size]
    batch = batch.fillna('')
    for id, row in batch.iterrows():
        rawtext = f"Item SKU: {row['item_sku']} Item Name: {row['item_name']} Short Description: {row['short_description']} Description: {row['description']} Brand: {row['brand']} Price: {row['unit_price']} Category: {row['category']} "
        print(row['item_sku'])
        #translated_text = translate_lang(rawtext)
        llmtexts.append(rawtext)
        doc = Document(page_content=rawtext, metadata=row.to_dict())
        docs.append(doc)
        #print(llmtexts)
    batch['llmtext'] = llmtexts

    #batch.to_csv(f"llm_product_{start}.csv", encoding='utf-8', index=False)

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")


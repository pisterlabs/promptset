import time
import os
import cassio
import csv
import pandas as pd
import openai

from test_pattern import OpenAITestPattern
from cassio.config import check_resolve_session


cassio.init(
    token=os.environ['ASTRA_DB_APPLICATION_TOKEN'],
    database_id=os.environ['ASTRA_DB_DATABASE_ID'],
    keyspace=os.environ.get('ASTRA_DB_KEYSPACE'),
)

def translate_lang(query):
    message_objects = []
    message_objects.append({"role":"user",
                        "content": "You are a electronics ecommerce store. The product inventory contains some mix of khmer and english content. Translate the complete text to English :'" +  query + "'"})
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=message_objects
    )
    text_in_en = completion.choices[0].message.content
   
    return text_in_en

#with open('Inventory.csv', 'r') as file:
#reader = csv.reader(file)
#headers = next(reader)
filename = 'Inventory.csv'
df = pd.read_csv(filename)
llmtexts = []
start = 5000
batch_size = 2000
for i in range(start, start+batch_size, batch_size):
    print(f"Processing {i} to {i+batch_size} llm texts")
    batch = df[i:i+batch_size]
    for id, row in batch.iterrows():
        rawtext = f"Item Code: {row['item_code']} Item Name: {row['item_name']} Description: {row['description']} Available: {row['availability']} Available: {row['availability']} Price: {row['price']} "
        print(row['item_code'])
        #translated_text = translate_lang(rawtext)
        llmtexts.append(rawtext)
        print(llmtexts)
    batch['llmtext'] = llmtexts
    #batch.to_csv(f"llm_product_{start}.csv", encoding='utf-8', index=False)
    test_pattern = OpenAITestPattern(session=check_resolve_session(None), model_name='text-embedding-ada-002',
                                api_key=os.environ['OPENAI_API_KEY'],
                                keyspace='ptc_ai_assistant',
                                table_name='ptc_inventory')
    # To make Product ID as document ID, use `adds_texts` instead of `from_documents`
    vstore = test_pattern.vectore_store()
    print(f"Adding {i} to {i+batch_size} vector store")
    if start == 0:
        vstore.clear()
    vstore.add_texts(texts=llmtexts,
                    metadatas=batch.to_dict(orient='records'),
                    ids=batch['item_code'].to_list())



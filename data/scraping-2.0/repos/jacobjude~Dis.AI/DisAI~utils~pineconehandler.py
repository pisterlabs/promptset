import pinecone
import openai
import asyncio
from concurrent.futures import ThreadPoolExecutor
import discord
import logging

from time import perf_counter
from config import PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_API_KEY

index = pinecone.Index("disai")
embed_model = "text-embedding-ada-002"
window = 1
stride = 1
batch_size=10

logger = logging.getLogger(__name__)

def create_embedding(texts, embed_model):
    return openai.Embedding.create(input=texts, engine=embed_model)

def upsert_vectors(to_upsert, namespace):
    index.upsert(vectors=to_upsert, namespace=namespace)


async def upsert_data(data, namespace, batch_number, window=1, stride=1, batch_size=10, type="chatbot", message_to_edit=None):
    """https://docs.pinecone.io/docs/gen-qa-openai to learn more about this function"""
    executor = ThreadPoolExecutor()
    if type == "chatbot":
        secondary_listing = "role"
        data_str = ""
    elif type == "PDF":
        secondary_listing = "(Page:"
        data_str = "PDF"
    elif type == "Video":
        secondary_listing = "(Timestamp:"
        data_str = "video"
    elif type == "lorebook":
        secondary_listing = "entry"
        data_str = ""
    new_data = []
    join_str = " "
    for i in range(0, len(data), stride): # for each {role, content} in data (skip by stride amount)
        await asyncio.sleep(0)
        i_end = min(len(data), i+window) 
        text_values = [item['content'] for item in data[i:i_end]]
        stamp = f"{secondary_listing} {data[i]['location']})" if (type != "chatbot" and type != "lorebook") else ""
        text = f"{stamp}{join_str.join(text_values)}{join_str}"
        # create the new merged dataset
        # join the text from the window together in one paragraph.
        new_data.append({
            'text': text,
            'id': str(i + batch_number),
        })
    # at this point, new_data is a list of text windows. now, prepare to upsert in batches of text windows.
    threshhold = 0
    for i in range(0, len(new_data), batch_size):
        
        await asyncio.sleep(0.02)
        percent = i / len(new_data) * 100
        print(percent)
        if message_to_edit and percent > threshhold:
            await message_to_edit.edit(embed=discord.Embed(title=f"Processing {data_str}...",  description=f"{percent:.2f}% Processed\n\n(This may take a while for large data...)", color=discord.Colour.blue()))
            threshhold += 20
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        try:
            res = await asyncio.get_event_loop().run_in_executor(executor, create_embedding, texts,embed_model)
        except Exception as e:
            done = False
            while not done:
                await asyncio.sleep(5)
                try:
                    res = await asyncio.get_event_loop().run_in_executor(executor, create_embedding, texts,embed_model)
                    done = True
                except Exception as e:
                    logger.error(f"Ratelimited: {e}")
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadatav
        meta_batch = [{
            'text': x['text'],
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        await asyncio.get_event_loop().run_in_executor(executor, upsert_vectors, to_upsert, namespace)
    return len(new_data) + batch_number + 1
    
def search_pinecone(query, namespace):
    try:
        res = openai.Embedding.create(
            input=[query],
            engine=embed_model
        )
        
        res = index.query(res['data'][0]['embedding'], top_k=3, include_metadata=True, namespace=namespace)
        return [match['metadata']['text'] for match in res['matches']]

    except Exception as e:
        logger.error(f"search pinecone err: {e}")
        return []

async def delete_namespace(namespace):
    index.delete(deleteAll='true', namespace=namespace)
    

def delete_namespace_nonasync(namespace):
    index.delete(deleteAll='true', namespace=namespace)

if __name__ == "__main__":
    while True:
        namespace = input("namespace:")
        delete_namespace_nonasync(namespace)
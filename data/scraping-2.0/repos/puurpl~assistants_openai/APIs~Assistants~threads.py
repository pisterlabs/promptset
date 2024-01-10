from openai import OpenAI

async def create_thread():
    client = OpenAI()
    thread = await client.beta.threads.create()
    return(thread)

async def retrieve_thread(thread_id):
    client = OpenAI()
    thread = await client.beta.threads.retrieve(thread_id)
    return(thread)

async def modify_thread(thread_id, metadata):
    client = OpenAI()
    thread = await client.beta.threads.update(
    thread_id=thread_id,
    metadata=metadata, # metadata is a map
    )
    return(thread)

async def delete_thread(thread_id):
    client = OpenAI()
    thread = await client.beta.threads.delete(thread_id)
    return(thread)

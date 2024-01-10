from openai import OpenAI

async def create_message(thread_id, role, content):
    client = OpenAI()
    message = await client.beta.threads.messages.create(
    thread_id=thread_id,
    role=role,
    content=content,
    )
    return(message)

async def list_messages(thread_id, order, limit):
    client = OpenAI()
    messages = await client.beta.threads.messages.list(
    thread_id=thread_id,
    order=order,
    limit=limit,
    )
    return(messages)

async def retrieve_message(thread_id, message_id):
    client = OpenAI()
    message = await client.beta.threads.messages.retrieve(
    thread_id=thread_id,
    message_id=message_id,
    )
    return(message)

async def modify_message(thread_id, message_id, content):
    client = OpenAI()
    message = await client.beta.threads.messages.update(
    thread_id=thread_id,
    message_id=message_id,
    metadata=content, # content is a map
    )
    return(message)

async def list_message_files(thread_id, message_id, order, limit):
    client = OpenAI()
    message_files = await client.beta.threads.messages.files.list(
    thread_id=thread_id,
    message_id=message_id,
    order=order,
    limit=limit,
    )
    return(message_files)

async def retrieve_message_file(thread_id, message_id, file_id):
    client = OpenAI()
    message_file = await client.beta.threads.messages.files.retrieve(
    thread_id=thread_id,
    message_id=message_id,
    file_id=file_id,
    )
    return(message_file)












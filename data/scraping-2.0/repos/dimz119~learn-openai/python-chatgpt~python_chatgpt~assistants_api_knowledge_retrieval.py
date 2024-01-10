import time
from openai import OpenAI

# make sure you have OPENAI_API_KEY environment variable with API key
# export OPENAI_API_KEY=""

client = OpenAI()

file = client.files.create(
            file=open("sample/event_time.txt", "rb"),
            purpose='assistants')

print(file.id)
"""
file-Vd3EP3i8VSDEmuBSaUjWlL4X
"""

print(file)
"""
FileObject(id='file-Vd3EP3i8VSDEmuBSaUjWlL4X', bytes=137, created_at=1704437660, filename='event_time.txt', object='file', purpose='assistants', status='processed', status_details=None)
"""

print(client.files.list())
"""
SyncPage[FileObject](data=[
    FileObject(
        id='file-Vd3EP3i8VSDEmuBSaUjWlL4X', 
        bytes=137, 
        created_at=1704437660, 
        filename='event_time.txt', 
        object='file', 
        purpose='assistants', 
        status='processed', 
        status_details=None), 
    ...
    ], object='list', has_more=False)
"""

# Add the file to the assistant
assistant = client.beta.assistants.create(
                instructions="You are a Superduper Hamburger customer support chatbot. Use your knowledge base from txt file to best respond to customer queries.",
                model="gpt-4-1106-preview",
                tools=[{"type": "retrieval"}],
                file_ids=[file.id])

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="What are the dates for Superduper Hamburger 2024 event period?",
            file_ids=[file.id])

run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id)

while True:
    run_resp = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    if run_resp.status == 'completed':
        break
    print(f"Waiting ... {run_resp.status}")
    time.sleep(3)

messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc",
                after=message.id)

for thread_message in messages.data:
    print(thread_message.content[0].text.value)
    print(thread_message)

"""
The dates for the Superduper Hamburger 2024 event period are from January 1st to January 15th, 
and March 15th to March 30th, 2024. During these periods, there will be huge discounts available.

ThreadMessage(
    id='msg_xvuWTj9UvMZL6SDvycAz2gu7',
    assistant_id='asst_dO5Eo41o2VVIlJm3wnTAsFkj',
    content=[
        MessageContentText(
            text=Text(annotations=[], value='The Superduper Hamburger 2024 event period is from January 1st to January 15th, and from March 15th to March 30th. During these periods, there will be huge discounts.'), type='text')], created_at=1704439806, file_ids=[], metadata={}, object='thread.message', role='assistant', run_id='run_8cegfLreRPQfMwtXqA8eElEH', thread_id='thread_qBRNNFQTHNvnqLfW80xUdWRo')
"""

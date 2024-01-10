from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import asyncio

app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    thread_id: str | None = None

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

async def send_message_get_reply(message, thread_id=None):
    print("Starting to send message...")
    client = openai.Client()
    assistant_id = "asst_CjvyFIeraCLKB8NTAqF0FhqG"

    start_time = asyncio.get_event_loop().time()
    if thread_id is None:
        print("Creating a new thread...")
        thread = await asyncio.to_thread(client.beta.threads.create)
        thread_id = thread.id

    print(f"Sending message to thread {thread_id}...")
    await asyncio.to_thread(client.beta.threads.messages.create, thread_id=thread_id, role="user", content=message)

    print("Creating run...")
    run = await asyncio.to_thread(client.beta.threads.runs.create, thread_id=thread_id, assistant_id=assistant_id)

    async def check_run_status(run_id):
        try:
            run_status = await asyncio.to_thread(client.beta.threads.runs.retrieve, thread_id=thread_id, run_id=run_id)
            return run_status.status
        except Exception as e:
            print(f"Error retrieving run: {e}")
            return None

    while True:
        run_status = await check_run_status(run.id)
        print(f"Run status: {run_status}")
        if run_status == 'completed':
            break
        elif run_status in ['failed', 'cancelled']:
            print("Run failed or was cancelled")
            return None, thread_id
        await asyncio.sleep(0.5)

    print("Fetching messages...")
    messages = await asyncio.to_thread(client.beta.threads.messages.list, thread_id=thread_id)
    last_message = next((m.content[0].text.value for m in messages.data if m.role == "assistant" and m.content), None)

    end_time = asyncio.get_event_loop().time()
    print(f"Total time taken: {end_time - start_time} seconds")
    return last_message, thread_id

@app.post("/chat/")
async def chat(request: MessageRequest):
    print(f"Received request: {request}")
    response, thread_id = await send_message_get_reply(request.message, request.thread_id)
    if response is None:
        print("Error: Failed to get a response from the assistant")
        raise HTTPException(status_code=500, detail="Failed to get a response from the assistant")
    return {"response": response, "thread_id": thread_id}

# uvicorn.run(app, host="0.0.0.0", port=8000)

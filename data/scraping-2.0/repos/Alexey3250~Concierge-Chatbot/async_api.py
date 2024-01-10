from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import asyncio

app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    
openai_api_key = os.getenv("OPENAI_API_KEY")
summariser_assistant_id = "asst_kCSrKaHjh589gbKr2fphQ93T"

@app.post("/summarise/")
async def summarise(request: MessageRequest):
    print("Starting summarisation process...")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=openai_api_key)

    # Run blocking operations in a background thread
    thread = await asyncio.to_thread(client.beta.threads.create)
    print(f"Thread created with ID: {thread.id}")

    await asyncio.to_thread(client.beta.threads.messages.create, thread_id=thread.id, role="user", content=request.message)
    print("Message added to the thread.")

    run = await asyncio.to_thread(client.beta.threads.runs.create, thread_id=thread.id, assistant_id=summariser_assistant_id)
    print(f"Run started with ID: {run.id}")

    async def check_run_status(thread_id, run_id):
        try:
            run = await asyncio.to_thread(client.beta.threads.runs.retrieve, thread_id=thread_id, run_id=run_id)
            print(f"Run status: {run.status}")
            return run.status
        except Exception as e:
            print(f"Error retrieving run status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    start_time = asyncio.get_event_loop().time()
    while True:
        if asyncio.get_event_loop().time() - start_time > 7:
            print("Run did not complete in time.")
            raise HTTPException(status_code=500, detail="Run did not complete in time.")
        
        status = await check_run_status(thread.id, run.id)
        if status == 'completed':
            print("Run completed successfully.")
            break
        elif status in ['failed', 'cancelled']:
            print(f"Run {status}.")
            raise HTTPException(status_code=500, detail=f"Run {status}")
        await asyncio.sleep(0.5)

    messages = await asyncio.to_thread(client.beta.threads.messages.list, thread_id=thread.id)
    
    for message in messages.data:
        if message.role == "assistant" and message.content:
            text_content = message.content[0].text
            if text_content:
                print("Returning the assistant's response.")
                return {"assistant_response": text_content.value}

    print("No response from the assistant.")
    raise HTTPException(status_code=500, detail="No response from the assistant")

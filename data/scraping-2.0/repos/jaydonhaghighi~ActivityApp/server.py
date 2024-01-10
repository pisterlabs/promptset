from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import uvicorn

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = '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'

chunks_store = {}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(data: GenerateRequest):
    prompt = data.prompt
    
    print(f"Received request for prompt: {prompt}")
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 
            """
                You will provide unique, creative, and exciting activities for people based on the parameters given by the user.
                (Every activity generated should adhere to the format below exactly)

                FORMAT:

                Activity:
                Title of the activity goes here

                Introduction:
                Insert a detailed introduction here

                Materials:
                (Maximum 5 short materials)
                - Material 1
                - Material 2
                - Material 3
                - Material 4
                - Material 5
                
                Instructions:
                (Maximum 5 instructions 1 sentence each)
                1. Instruction 1
                2. Instruction 2
                3. Instruction 3
                4. Instruction 4
                5. Instruction 5

                Note:
                Insert a short note here
            """},
            {"role": "user", "content": prompt}
        ]
    )

    chunks = response.choices[0].message['content'].strip().split()
    request_key = str(hash(prompt))
    chunks_store[request_key] = chunks
    first_chunk = chunks_store[request_key].pop(0) if chunks_store[request_key] else None  # Use None as a sentinel value
    
    if first_chunk is None:
        return {'response': '', 'message': 'No more chunks'}
    
    print(f"Returning first chunk: {first_chunk} for request key: {request_key}")
    
    return {
        'response': first_chunk,
        'request_key': request_key
    }

@app.get("/next_chunk/{request_key}")
async def next_chunk(request_key: str):
    print(f"Request for next chunk with key: {request_key}")

    if request_key not in chunks_store or not chunks_store[request_key]:
        return {'response': '', 'message': 'No more chunks'}  # Return a specific message to indicate end

    next_chunk = chunks_store[request_key].pop(0)
    print(f"Returning chunk: {next_chunk} for request key: {request_key}")
    return {'response': next_chunk}
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://blank-to-bard-frontend-bi2gia7neq-ez.a.run.app"],  # Change this to the actual URL of your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
# Define a root `/` endpoint that shows all the available endpoints
async def root():
    return {"message": "Welcome to Whisper STT API!"}

@app.post("/transcribe/{language}")
async def transcribe_audio(audio: UploadFile = File(...), language: str = "en"):
    # Save temporary audio file
    with open("temp_audio.mp3", "wb") as buffer:
        buffer.write(await audio.read())

     # Transcribe the audio
    with open("temp_audio.mp3", "rb") as audio_file:
        result = openai.Audio.transcribe("whisper-1", audio_file, language=language)

    return {"transcription": result["text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
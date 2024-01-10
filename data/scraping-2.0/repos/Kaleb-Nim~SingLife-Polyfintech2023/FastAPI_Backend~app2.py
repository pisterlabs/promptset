from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import requests
from openai import AsyncAzureOpenAI
from PineconeUtils.Queryer import PineconeQuery
from dotenv import load_dotenv
import os
from llm.chains import generate_video,relevantDocumentFilter,generate_video_agent
from utils import formatQuery, parse_json_output
from pydantic import BaseModel
from typing import Optional
import random
from elevenlabs import generate as generate_voice, set_api_key, voices
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from urllib.parse import quote
import json
import whisper_timestamped as whisper

# Load variables from the .env file
load_dotenv(".env")
from fastapi.middleware.cors import CORSMiddleware

# Set the openai api key
client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# PINECONE
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# HUGGINGFACE
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ELEVENLABS
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
set_api_key(ELEVENLABS_API_KEY)

# AZURE
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
connection_string = AZURE_STORAGE_KEY
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
account_name = connection_string.split(";")[1].split("=")[1]

# PEXELS
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

pineconeQuery = PineconeQuery(PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def generate_uuid():
    # Generate a random UUID
    uuid = "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}".format(
        random.getrandbits(32),
        random.getrandbits(16),
        (random.getrandbits(12) & 0x0fff) | 0x4000,
        (random.getrandbits(12) & 0x3fff) | 0x8000,
        random.getrandbits(48)
    )
    return uuid

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("Request for index page received")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    file_name = "favicon.ico"
    file_path = "./static/" + file_name
    return FileResponse(
        path=file_path, headers={"mimetype": "image/vnd.microsoft.icon"}
    )


@app.post("/hello", response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print("Request for hello page received with name=%s" % name)
        return templates.TemplateResponse(
            "hello.html", {"request": request, "name": name}
        )
    else:
        print(
            "Request for hello page received with no name or blank name -- redirecting"
        )
        return RedirectResponse(
            request.url_for("index"), status_code=status.HTTP_302_FOUND
        )


class UserInfo(BaseModel):
    name: str
    age: Optional[int] = None
    concerns: str
    needs: Optional[str] = None
    lifestyle: Optional[str] = None


# Test POSTMAN query request
@app.post("/query")
async def query(UserInfo: UserInfo):
    """Sample UserInfo must contain the following keys:
    name: str,
    age: Optional[int] = None,
    concerns:str,
    needs:Optional[str] = None,
    lifestyle:Optional[str] = None,
    """
    query_dict = formatQuery(UserInfo)
    relevant_documents = pineconeQuery.query(query=query_dict["pinecone_query"])

    # new_relevant_documents = await relevantDocumentFilter(relevant_documents,query_dict["user_query"])

    relevant_documents_str = pineconeQuery.concatDocuments(relevant_documents)
    sources = pineconeQuery.extractDocumentSources(relevant_documents)
    
    # Filter out the relevant documents

    # Run the LLM for video generation
    video_script = await generate_video_agent(
        relevant_documents=relevant_documents_str, query=query_dict["user_query"]
    )
    video_script_json = parse_json_output(video_script)



    return {
        "query": query_dict["user_query"],
        "relevant_documents": relevant_documents,
        "video_script": video_script_json,
        "sources": sources,
        # "new_relevant_documents":new_relevant_documents
    }
@app.post("/query0")
async def query_old(UserInfo: UserInfo):
    """Sample UserInfo must contain the following keys:
    name: str,
    age: Optional[int] = None,
    concerns:str,
    needs:Optional[str] = None,
    lifestyle:Optional[str] = None,
    """
    query_dict = formatQuery(UserInfo)
    relevant_documents = pineconeQuery.query(query=query_dict["pinecone_query"])

    # new_relevant_documents = await relevantDocumentFilter(relevant_documents,query_dict["user_query"])

    relevant_documents_str = pineconeQuery.concatDocuments(relevant_documents)
    sources = pineconeQuery.extractDocumentSources(relevant_documents)
    
    # Filter out the relevant documents

    # Run the LLM for video generation
    video_script = await generate_video(
        relevant_documents=relevant_documents_str, query=query_dict["user_query"]
    )
    video_script_json = parse_json_output(video_script)



    return {
        "query": query_dict["user_query"],
        "relevant_documents": relevant_documents,
        "video_script": video_script_json,
        "sources": sources,
        # "new_relevant_documents":new_relevant_documents
    }

@app.post("/generateMusic")
async def generateMusic():
    music_style = ["slow pace loopable advertisement music"]
    random_music = music_style[random.randint(0, len(music_style) - 1)]
    API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query(
        {
            "inputs": random_music,
        }
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    blob_name = f"music_{current_time}_{generate_uuid()}.wav"

    blob_client = blob_service_client.get_blob_client(container="music", blob=blob_name)

    blob_client.upload_blob(audio_bytes, overwrite=True)
    blob_uri = blob_client.url
    return {"music": blob_uri}


class VideoBody(BaseModel):
    scene: list[str]


@app.post("/generateVideo")
async def generateVideo(VideoBody: VideoBody):
    blob_storage_array = []
    for scene in VideoBody.scene:
        response = requests.get(
            f"https://api.pexels.com/videos/search?query={quote(scene)}&per_page=1&orientation=landscape&size=medium",
            headers={"Authorization": PEXELS_API_KEY},
        )
        responseContent = json.loads(response.content)
        if len(responseContent["videos"]) > 0:
            print(responseContent)
            videoLink = responseContent["videos"][0]["video_files"][0]["link"]
            videoResponse = requests.get(videoLink)
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            blob_name = f"{scene}_{current_time}_{generate_uuid()}.mp4"
            blob_client = blob_service_client.get_blob_client(
                container="video", blob=blob_name
            )
            blob_client.upload_blob(videoResponse.content, overwrite=True)
            blob_uri = blob_client.url
            blob_storage_array.append(blob_uri)
    return {"video": blob_storage_array}


class VoiceBody(BaseModel):
    subtitles: list[str]


@app.post("/generateVoice")
async def generateVoice(VoiceBody: VoiceBody):
    allText = "\n".join(VoiceBody.subtitles)
    voiceList = voices()
    selectedVoice = random.choices(voiceList)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # audio = generate_voice(text=allText, voice=selectedVoice[0])
    # blob_name = f"audio_{current_time}_{generate_uuid()}.mp3"
    # blob_client = blob_service_client.get_blob_client(container="audio", blob=blob_name)
    # blob_client.upload_blob(audio, overwrite=True)
    # blob_uri = blob_client.url
    blob_uri = "https://singen.blob.core.windows.net/audio/audio_2023-11-13_23-48-35_a1d9accc6ed945f195ff1daa2635d50e.mp3"

    # SRT FILE
    whisper_audio = whisper.load_audio(blob_uri)
    model = whisper.load_model("base")

    result = whisper.transcribe(model, whisper_audio, language="en")

    srt_file = ""

    for i, segment in enumerate(result["segments"]):
        start, end = segment["start"], segment["end"]
        srt_file += f"{i + 1}\n00:00:{str(int(start)).replace('.', ',')} --> 00:00:{str(int(end)).replace('.', ',')}\n{segment['text'].strip()}\n"
    print(srt_file)
    srt_blob_name = f"subtitles_{current_time}_{generate_uuid()}.srt"
    srt_blob_client = blob_service_client.get_blob_client(container="srt", blob=srt_blob_name)
    srt_blob_client.upload_blob(srt_file, overwrite=True)
    srt_blob_uri = srt_blob_client.url
    return {"audio": blob_uri, "srt_file": srt_blob_uri}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from pathlib import Path
from fastapi.responses import StreamingResponse
from time import sleep
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import(
    HumanMessage, 
    SystemMessage
)
from openai import OpenAI


router = APIRouter(
    prefix="/speach-and-text",
    tags=["Speach and Text"]
)

@router.post("/speech-into-text/")
def speech_into_text(fileName: str = "audio1"):
    client = OpenAI()
    audio_path = Path(f'./API/Audio/{fileName}.mp3')        
    audio_file = open(audio_path, "rb")

    transcript = OpenAI().audio.transcriptions.create(
        model = "whisper-1",
        file = audio_file,
        response_format = "text",
    )
    return transcript


@router.post("/text-into-speech/")
def text_into_speech(text: str = "Tell me about yourself. Something like your name, where you are from, what you do, and what you are passionate about."):
    audioResponse = OpenAI().audio.speech.create(
            model = "tts-1",
            voice= "nova",
            input = text,
        )
    
    audioResponse.stream_to_file("./API/Audio/TextToAudio.mp3")
    return True   
    
@router.post("/audio-prompt/")
def audio_prompt(fileName: str = "audio1"):
    transcript = speech_into_text(fileName)
    llmResponse = call_openAi_llm(transcript)
    audioResponse = text_into_speech(llmResponse)
    return audioResponse

def call_openAi_llm(text: str):
    messages = [
            SystemMessage(content='You are an expert in not answering questions and changing the subject.The customer should never get a direct answer. Do your best to input irony to the answer.Avoid ansering it at all cost.Always answer in english.'),
            HumanMessage(content=f'Provide the most absurd answer or comment you can think of to the the following text:\n TEXT: {text}')
        ]
    llm = ChatOpenAI(model_name="gpt-3.5-turbo" ,temperature=0.7)
    output = llm(messages)
    return output.content
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.messages import HumanMessage, SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
from langchain import VectorDBQA
from dotenv import load_dotenv
from utils.prompts import *
from typing import Optional
import moviepy.editor as mp
import pinecone
import whisper
import os

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = "us-east4-gcp"
embed_model = "text-embedding-ada-002"

# Pinecone Setup
pinecone.init(api_key = PINECONE_API_KEY, environment = PINECONE_ENV)
index_name = 'kb-builder' # Set the index name in pinecone first


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(model="gpt-4-1106-preview",temperature=0, openai_api_key=OPENAI_API_KEY)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["."], chunk_size=1000, chunk_overlap=25
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
summaries = []
transcriber_model_sizes = ["tiny.en","base.en","small.en","medium","large-v3"]
summary_lengths = {"Short": "5-10 WORD","Medium": "30-50 WORD","Long": "75-125 WORD"}

def transcribe(content, content_type, model_size: Optional[str] = "tiny.en"):
    global transcript
    print(f"Content type is {content_type}")
    if content_type == "Youtube":
        video_id = content.replace('https://www.youtube.com/watch?v=', '')
        result = YouTubeTranscriptApi.get_transcript(video_id)
        transcript=''
        for x in result:
            sentence = x['text']
            transcript += f' {sentence}\n'

    elif content_type == "Article":
        print("Article")
        transcript="Article transcript goes here"

    elif content_type == "Tweet":
        print("Tweet")
        transcript="Tweet transcript goes here"

    elif content_type == "Video":
        transcriber_model = whisper.load_model(transcriber_model_sizes[model_size])
        # First we must convert the video to Audio
        mp.VideoFileClip(content).audio.write_audiofile("data/Convertedaudio.wav")
        output = transcriber_model.transcribe("data/Convertedaudio.wav")
        transcript = output["text"].strip()

    elif content_type == "Audio":
        transcriber_model = whisper.load_model(transcriber_model_sizes[model_size])
        # Extract the audio data from the tuple
        audio_data = content[1] if isinstance(content, tuple) else content
        mp.AudioFileClip(content).write_audiofile("data/audio.wav")
        output = transcriber_model.transcribe(audio_data)
        transcript = output["text"].strip()

    else:
        return f'Error, content type({content_type}) is not found!'

    return transcript


def add_to_knowledge_base(summary):
    print("Adding summary to vector KB")
    global texts
    texts = text_splitter.create_documents([summary])

    if index_name not in pinecone.list_indexes():
        print("Index does not exist: ", index_name)

    try:
        db = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name = index_name)
        global qa
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db)
        return "Successfully Added to Vector DB!"
    
    except SystemError:
        return "Error: Could not add summary to DB"

def summarise():
    messages = [
        SystemMessage(content=spr_prompt),
        HumanMessage(content=f"Here is a transcript, please summarise it: {transcript}"),
    ]
    
    summary = chat.invoke(messages).content

    return summary


def question_answer(question: str):
    try:
        return qa.run(question)
       
    except NameError:
        return "Error: Ensure summary has been added to db first!"

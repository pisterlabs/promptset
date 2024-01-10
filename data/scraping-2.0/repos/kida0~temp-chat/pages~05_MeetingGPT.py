# 미팅 영상의 음성만 추출한 후 Whisper를 통해 음성을 텍스트로 요약하여 사용

import enum
import os
import math
import subprocess
from pydub import AudioSegment
import streamlit as st
import openai
import glob

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.storage.file_system import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🦄",
)

st.title("MeetingGPT")
st.markdown(
    """
    Welcome to MeetingGPT, upload a video and I will give you a transcript, 
    a summary and a chatbot to ask any questions about it.
    
    Get started by uploading a video file in the sidebar.
    """
)

has_transcript = os.path.exists("./.cache/podcast.txt")

llm = ChatOpenAI(
    temperature=0.1
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    final_transcript = ""
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1", 
                audio_file
            )
            text_file.write(transcript["text"])
            
            
@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)
    
    
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3")


with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])
    
if video:
    video_path = f"./.cache/{video.name}"
    chunks_folder = "./.cache/chunks"
    audio_path = video_path.replace("mp4", "mp3")
    transcript_path = audio_path.replace("mp3", "txt")
        
    with st.status("Loading video...") as status:
        video_content = video.read()
    with open(video_path, "wb") as f:
        f.write(video_content)
    
    status.update(label="Extracting audio...")
    extract_audio_from_video(video_path)
    
    status.update(label="Cutting audio...")
    cut_audio_in_chunks(audio_path, 10, chunks_folder)
    
    status.update(label="Transcribing audio...")
    transcribe_chunks(chunks_folder, transcript_path)
            
    transcript_tab, summary_tab, qna_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
            
    with summary_tab:
        start = st.button("Generate summary")
        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)
            
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                    다음 문장을 간결하게 요약해줘:
                    "{text}"
                    간결한 요약 내용:
                """
            )
            
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            
            summary = first_summary_chain.invoke({
                "text": docs[0].page_content
            })
            
            refine_prompt = ChatPromptTemplate.from_template(
                """
                    Your job is to produce a final summary.
                    We have provided an existsing summary up to a
                    certain point: {existing_summary}
                    We have the opporunity to reifne the existing summary
                    (only if needed) with some more context below.
                    ----------
                    {context}
                    ----------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                """
            )
            
            refine_chain = refine_prompt | llm | StrOutputParser()
            
            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context": doc.page_content,
                    })
            
            st.write(summary)
            
    with qna_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("블리자드의 현재 상황에 대해 알려줘")
        st.write(docs)
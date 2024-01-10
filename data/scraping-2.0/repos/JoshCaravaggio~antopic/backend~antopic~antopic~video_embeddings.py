import errno
import multiprocessing
import os
import yt_dlp
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.vectorstores import Pinecone
from langchain import OpenAI
import pinecone
import subprocess

# Progress hook for download 
def yt_dl_progress_hook(d):
    if d['status'] == 'finished':
        print(f"\nVideo downloaded successfully to: {d['filename']}")
    if d['status'] == 'downloading':
        progress = d['_percent_str']
        print(f"Downloading progress: {progress}", end='\r')

# download_youtube_video
def download_youtube_video(url, content_id, output_dir="./audio/"):
    content_filename = output_dir + content_id
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': content_filename,
        'progress_hooks': [yt_dl_progress_hook],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        print(f"Error: {e}")
    return content_filename + '.mp3'


def transcribe_local_audio(filepath, id, output_dir="./transcribed/"):

    openai.api_key = os.environ["OPENAI_API_KEY"]

    with open(filepath, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
        print("Successfully transcribed ")
        write_path = output_dir + os.path.basename(id) + ".txt"
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        print("Writing transcript to " + write_path)
        with open(write_path, "w+") as transcript_file:
            transcript_file.write(transcript.text)
        print("Transcript saved to " + write_path)
    return write_path


def load_and_split_text(file_path):
    print('Loading text')
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    print('Splitting text')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    split = text_splitter.split_documents(docs)
    return split


def load_and_split_pdf(file_path):

    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    split = text_splitter.split_documents(docs)
    return split


def create_embeddings(docs, id):
    print('Creating embeddings')
    model = OpenAIEmbeddings(model="text-embedding-ada-002",
                             chunk_size=1000)
    contents = list(map(lambda d: d.page_content, docs))
    print('Embedding docs')
    embeddings = model.embed_documents(contents)

    if os.environ['PERSIST_EMBEDDINGS']:
        write_path = persist_embeddings(id, embeddings)
        print(f'Embeddings persisted to {write_path}')
    return embeddings


def persist_embeddings(id, embeddings):
    print('Persisting embeddings')
    write_path = f'./embeddings/{id}.txt'
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    embedding_persist = ''
    for item in embeddings:
        embedding_persist += ''.join(str(item))
    with open(write_path, 'w') as f:
        f.write(embedding_persist)
    return write_path


def store_embeddings(docs, id, topic_id):
    model = OpenAIEmbeddings(model="text-embedding-ada-002",
                             chunk_size=1000)
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
                  environment=os.environ['PINECONE_ENV'])
    index = pinecone.Index(os.environ['PINECONE_INDEX'])
    vectorstore = Pinecone(index, model.embed_query,
                           "text", namespace=topic_id)
    ids = vectorstore.add_documents(docs)
    return ids


def store_video_embeddings(url, id, topic_id):
    print(f'Donwnloading {url} with content ID {id} for topic {topic_id}')
    audio_filepath = download_youtube_video(url, id)
    print(f'Chunking {audio_filepath}')
    chunks = split_mp3(file_path=audio_filepath)
    print(f'Transcribing chunked {audio_filepath}')
    doc_ids = []
    for chunk in chunks:
        print(f'Transcribing chunk {chunk}')
        transcript_file_path = transcribe_local_audio(chunk, id)
        docs = load_and_split_text(transcript_file_path)
        doc_ids.extend(store_embeddings(docs, id, topic_id))
    silentcleanup(id)
    return doc_ids


def store_text_embeddings(text_file_path, id, topic_id):
    docs = load_and_split_text(text_file_path)
    store_embeddings(docs, id, topic_id)


def store_pdf_embeddings(text_file_path, id, topic_id):
    docs = load_and_split_pdf(text_file_path)
    store_embeddings(docs, id, topic_id)


def store_audio_embeddings(audio_filepath, id, topic_id):
    print(f'Chunking {audio_filepath}')
    chunks = split_mp3(file_path=audio_filepath)
    print(f'Transcribing chunked {audio_filepath}')
    doc_ids = []
    for chunk in chunks:
        print(f'Transcribing chunk {chunk}')
        transcript_file_path = transcribe_local_audio(chunk, id)
        docs = load_and_split_text(transcript_file_path)
        doc_ids.extend(store_embeddings(docs, id, topic_id))
    silentcleanup(id)
    return doc_ids


def silentcleanup(content_id):
    print("Cleaning up files")
    try:
        print(f'Removing ./audio/{content_id}.mp3')
        os.remove(f'./audio/{content_id}.mp3')
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred
    try:
        print(f'Removing ./transcribed/{content_id}.txt')
        os.remove(f'./transcribed/{content_id}.txt')
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def split_mp3(file_path, chunk_size=25):
    chunks_dir = os.path.dirname(file_path)
    chunks_dir = f'{chunks_dir}/chunks'
    os.makedirs(chunks_dir, exist_ok=True)

    # Get the duration of the input MP3 file
    duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
    duration = float(subprocess.check_output(
        duration_cmd, shell=True, text=True).strip())

    # Calculate the number of chunks needed based on the duration and desired chunk size
    bitrate = (chunk_size * 8000) / duration
    chunk_duration = duration * \
        (chunk_size / (os.path.getsize(file_path) / (1024 * 1024)))

    # Split the MP3 file into chunks
    num_chunks = int(duration // chunk_duration) + 1
    output_chunks = []
    for i in range(num_chunks):
        split_chunk(file_path, chunks_dir, bitrate,
                    chunk_duration, output_chunks, i)

    return output_chunks


def split_chunk(file_path, chunks_dir, bitrate, chunk_duration, output_chunks, i):
    start_time = i * chunk_duration
    output_file = os.path.join(chunks_dir, f'chunk_{i}.mp3')
    ffmpeg_cmd = f'ffmpeg -y -i "{file_path}" -ss {start_time} -t {chunk_duration} -b:a {bitrate} "{output_file}"'
    subprocess.run(ffmpeg_cmd, shell=True)
    output_chunks.append(output_file)

import os
import pickle
import requests

import boto3
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from pydub import AudioSegment

from ..utils.audio import copy_audio 
from ..utils.transcribe import transcribe_file
from ..processors.whisper import WhisperProcessor
from ..documents.base import BaseDocument

PROCESSORS = {
    'whisper': WhisperProcessor 
}


class TranscriptDocument(BaseDocument):
    def __init__(self, name, uri, processor='whisper', bucket=None):
        self.name = name
        self.uri = uri
        self.bucket = bucket

        # Private attributes 
        self.text = "" 
        self.documents = [] 
        self.summary = None
 
        # Set the data directory and source document path
        self.datadir = f".{self.name}__transcript"
        self.source_path = f"{self.datadir}/audio/{self.name}.mp3" 
        self.processor = PROCESSORS[processor](self.datadir, self.source_path, bucket=bucket)
 
        # Initialize the LLM
        # TODO: add llm initialization params
        self.llm = ChatOpenAI(max_tokens=1000, temperature=0.2, model_name="gpt-3.5-turbo-16k")

        if not os.path.exists(self.datadir):
            self._mkdirs()
 
    def _mkdirs(self):
        folders = ['', 'audio', 'pickles', 'text']
        if not os.path.exists(self.datadir):
            for folder in folders:
                os.mkdir(os.path.join(self.datadir, folder))

    def split(self, chunk_size_mb=10):
        audio = AudioSegment.from_mp3(self.source_path)

        # Determine the number of chunks based on the size of the original file
        file_size_bytes = os.path.getsize(self.source_path)
        n = file_size_bytes // (chunk_size_mb * 1024 * 1024)
        print(f"Num chunks: {n}")

        # Calculate the length of each chunk in milliseconds
        chunk_length = len(audio) // n

        # Check for existing output folder and create if not exists
        folder_path = f'{self.datadir}/audio/chunks'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Split and save the chunks
        for i in range(n):
            chunk_path = os.path.join(folder_path, f'chunk_{i}.mp3')
            if not os.path.exists(chunk_path):
                start_time = i * chunk_length
                end_time = (i + 1) * chunk_length if i < n - 1 else len(audio)
                chunk = audio[start_time:end_time]

                chunk.export(chunk_path, format='mp3')
                print(f'Saved {chunk_path}')
            else:
                print(f"Chunk {chunk_path} exists, skipping")

    def download(self, sync=False):
        """ Downloads the audio file based on the URI and writes it to a local file """
        if not os.path.exists(self.source_path):
            copy_audio(self.uri, self.source_path)
        else:
            print("File exists locally, skipping download")

        if sync:
            self.sync()

    def process(self):
        """ Transcribes the audio and sets the text, documents, and summary of the document """
        self.split(chunk_size_mb=5)
        # Transcribe the audio
        self.processor.process()
        # Get the text
        text, chunk_text = self.processor.get_text() 
        self.text = text
        self.chunk_text = chunk_text
        # Set the documents
        self.set_documents()
        # self.sync()
 
    def set_transcript(self):
        transcribe_client = boto3.client('transcribe', region_name='us-east-1')
        transcript_url = transcribe_file(
            self.name,
            f's3://{self.bucket}/{self.datadir}/audio/{self.name}.mp3',
            transcribe_client
        )
        if transcript_url:
            res = requests.get(transcript_url)
            self.transcription_data = res.json()
            self.text = self.transcription_data['transcripts'][0]['transcript'] 
            # Write the text file of the transcript
            with open(f'{self.datadir}/text/{self.name}.txt', 'w') as f:
                f.write(self.text)
        else:
            print("Error transcribing audio")

    def set_documents(self, chunk_size=1024):
        """ Sets documents on the object """
        if not self.text:
            raise

        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=64,
            length_function=len,
        )
        chunks = text_splitter.split_text(self.text)
        self.documents = [Document(page_content=t) for t in chunks] 

    def summarize(self):
        """ Generates a map_reduce summary of the object """
        print("Summarizing transcript")
        summary_path = f'{self.datadir}/pickles/summary.pkl'

        if os.path.exists(summary_path):
            print("Summary exists, not generating")
            with open(summary_path, 'rb') as f:
                summary = pickle.load(f)
            self.summary = summary
        else:
            print("Summary doesn't exist, generating")
            if not self.documents:
                self.set_documents()

            # Create the summary template
            template = "You are an expert at summarizing city council meeting transcripts. Summarize everything but the general business and public hearing items."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)

            # Create the human template
            human_template = "{text}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            # Generate the overall prompt
            prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt],
            )

            # Create the chain and summarize
            chain = load_summarize_chain(self.llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
            summary = chain({"input_documents": self.documents}, return_only_outputs=True)
            self.summary = summary

            # Pickle the summary 
            with open(f'{self.datadir}/pickles/summary.pkl', 'wb') as f:
                pickle.dump(summary, f)

        return summary
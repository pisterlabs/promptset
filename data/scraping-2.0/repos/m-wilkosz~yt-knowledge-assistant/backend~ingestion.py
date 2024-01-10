import os
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from backend.consts import INDEX_NAME
from backend.core import summary_chain
import streamlit as st

load_dotenv()

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT_REGION'],
)

def merge_chunks(data):
    merged_data = []

    for i in range(0, len(data), 5):
        chunk = data[i:i+5]

        # calculate the start time and duration for the merged chunk
        start_time = chunk[0]['start']
        end_time = chunk[-1]['start'] + chunk[-1]['duration']
        duration = end_time - start_time

        # merge the text of all items in the chunk
        text = ' '.join([item['text'] for item in chunk])

        merged_data.append({
            'duration': duration,
            'start': start_time,
            'text': text
        })
    return merged_data

def ingest_cc(video_id):
    # get youtube video captions
    cc = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
    merged_cc = merge_chunks(cc)
    string_cc = str(merged_cc)

    # create temporary file from captions string, load it into document and split it
    with tempfile.NamedTemporaryFile(delete=True, mode='w', suffix='.txt') as temp_file:
        temp_file.write(string_cc)
        temp_file.flush()

        loader = TextLoader(temp_file.name)
        loaded_cc = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator='}')
        docs = text_splitter.split_documents(loaded_cc)

        # create summary
        summary_text_splitter = CharacterTextSplitter(
                                        chunk_size=15000,
                                        chunk_overlap=500,
                                        separator='}'
                                        )
        docs_for_summary = summary_text_splitter.split_documents(loaded_cc)
        summary = summary_chain(docs_for_summary)

    # create embeddings from docs and add them to vectorstore
    embeddings = OpenAIEmbeddings()

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    pinecone_nmspc = Pinecone(
        pinecone.Index(index_name=INDEX_NAME),
        embedding=embeddings,
        text_key='text',
        namespace=video_id
    )

    pinecone_nmspc.add_texts(
        texts,
        metadatas=metadatas,
        namespace=video_id,
    )

    print(f'added {len(docs)} vectors to pinecone vectorstore')

    return summary
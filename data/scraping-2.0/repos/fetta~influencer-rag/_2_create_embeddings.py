import json
import logging
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from vector_db import chroma_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

chroma = chroma_provider.get_chroma()


def process_transcript(file_path):
    logger.info(f"Processing file: {file_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

    with open(file_path, 'r') as file:
        data_list = json.load(file)

        for data in data_list:
            chunks = splitter.split_text(data['transcript'])

            # for doc_number, chunk in enumerate(chunks):
            #     print(f"- {doc_number}: '{chunk}' ({len(chunk.split())})")
            # break

            video_metadata = {
                "title": data['title'],
                "channel": data['channel'],
                "url": data['url'],
                "istitle": False,
                "file": file_path
            }

            video_title_metadata = video_metadata.copy()
            video_title_metadata['istitle'] = True

            # First chunk is the title, others follow
            texts = [data['title']] + chunks
            metadatas=[video_title_metadata] + [video_metadata] * (len(chunks))

            if len(config.channels) == 1:
                # Adds IDs, might be used to obtain related chunks;
                # temporarily works only with one channel defined in config.py
                ids=[str(0)] + list(map(lambda x: str(x), list(range(1, len(chunks) + 1))))
                chroma.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            else:
                chroma.add_texts(texts=texts, metadatas=metadatas)

            chroma.persist()
            print(f"Added {len(chunks)} embeddings for video '{data['title']}'")

def main():
    for channel in config.channels:
        transcript_file_path = f"{config.transcripts_dir_path}/{channel.handle}_transcripts.json"
        if os.path.exists(transcript_file_path):
            process_transcript(transcript_file_path)
            logger.info(f"Completed processing for file: {transcript_file_path}")
        else:
            logger.error(f"File: {transcript_file_path} does not exists")


if __name__ == "__main__":
    main()

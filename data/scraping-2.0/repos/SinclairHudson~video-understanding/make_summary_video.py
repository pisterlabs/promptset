from langchain.retrievers import BM25Retriever
import os
from langchain.schema import Document
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from moviepy.editor import VideoFileClip, concatenate_videoclips
import time


def compose_summary_video(clip_dir, clip_descriptions, query: str, k=30):
    documents = []
    df = pd.read_csv(clip_descriptions)
    description_columns = [col for col in df.columns if 'desc' in col]
    for index, row in df.iterrows():
        content = ' '.join([row[col] for col in description_columns])
        documents.append(Document(page_content=content, metadata={'file': row['clip_file']}))

    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={'k': k})
    start = time.time()
    results = retriever.get_relevant_documents(query)
    print(f"Retrieval took {time.time() - start} seconds")
    files = [result.metadata['file'] for result in results]

    clips = []
    for file in files:
        try:
            clip = VideoFileClip(os.path.join(clip_dir, file))
            clips.append(clip)
        except Exception as e:
            print(f"something up with file {file}")
            print(e)
    # clips = [VideoFileClip(os.path.join(clip_dir, file)) for file in files]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(f"summary_k={k}_prompt={query}.mp4")

if __name__ == '__main__':
    compose_summary_video("hockey_tmp", "descriptions/hockey_clip_descriptions.csv", "penalty")
    # compose_summary_video("soccer_tmp", "descriptions/soccer_clip_descriptions.csv", "penalty")
    # compose_summary_video("soccer_tmp", "descriptions/soccer_clip_descriptions.csv", "coach")



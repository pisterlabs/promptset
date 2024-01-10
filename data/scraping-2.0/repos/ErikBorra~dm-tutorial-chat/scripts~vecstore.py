import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS  # pip install faiss-gpu

rebuild_vecstore = True

load_dotenv('.env')

# create embeddings
embeddings = OpenAIEmbeddings()

# load the vectorstore
if rebuild_vecstore or not os.path.exists("faiss_index"):
    # *** Chunk size: key parameter ***
    chunks = 1500
    splits_new = []
    metadatas_new = []

    # Read the csv file
    new_ep = pd.read_csv("audio_transcription/episodes.csv", index_col=None)

    for ix in new_ep.index:

        # Get data
        title = new_ep.loc[ix, 'title']
        ep_number = int(new_ep.loc[ix, 'index'])
        episode_id = new_ep.loc[ix, 'id']

        file_path = 'audio_transcription/%s.txt' % str(episode_id)
        transcript = pd.read_csv(file_path, sep='\t', header=None)
        transcript.columns = ['links', 'time', 'chunks']

        # Clean text chunks
        transcript['clean_chunks'] = transcript['chunks'].astype(
            str).apply(lambda x: x.strip())
        links = list(transcript['links'])

        # Concatenate all chunks
        texts = transcript['clean_chunks'].str.cat(sep=' ')

        # Split them in chunk size for the vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunks,
                                                       chunk_overlap=100)
        splits = text_splitter.split_text(texts)
        # print(len(splits))

        # Metadata
        N = len(splits)
        bins = np.linspace(0, len(links)-1, N, dtype=int)
        sampled_links = [links[i] for i in bins]

        # Here we can add "link", "title", etc that can be fetched in the app
        metadatas = [{"source": link, "id": episode_id,
                      "link": link, "title": title} for link in sampled_links]
        # print(len(metadatas))

        # Append to output
        splits_new.append(splits)
        metadatas_new.append(metadatas)

    # Join the list of lists
    splits_all = []
    for sublist in splits_new:
        splits_all.extend(sublist)

    metadatas_all = []
    for sublist in metadatas_new:
        metadatas_all.extend(sublist)

    # create the vectorestore to use as the index
    db = FAISS.from_texts(
        texts=splits_all, embedding=embeddings, metadatas=metadatas_all)

    db.save_local("faiss_index")

else:
    db = FAISS.load_local("faiss_index", embeddings)

if not rebuild_vecstore:
    # expose this index in a retriever interface
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 2})
    print(retriever)

    # Build a QA chain
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    a = qa({"question": "What is a research browser and how to use it?", "verbose": True},
           return_only_outputs=True)
    print(a)
    # print(a["answer"])
    # print(a["sources"])

import csv
import pandas as pd
import numpy as np
from langchain.text_splitter import LatexTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')
model = model.to("cuda")

def chunked_tokens(text):
    latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = latex_splitter.create_documents([text])
    return docs


def get_embedding(text):
    vector_value = model.encode(text)
    return vector_value


def len_safe_get_embedding(text, average=True):
    try:
        chunk_embeddings = []
        chunk_lens = []
        for docs in chunked_tokens(text):
            chunk_embeddings.append(get_embedding(docs.page_content))
            chunk_lens.append(len(docs.page_content))

        if average:
            chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
            chunk_embeddings = chunk_embeddings.tolist()

    except Exception as e:
        print(f'Exception : {e}')
        return None

    return chunk_embeddings


def data_conversion():
    raw_content_vectors = []

    refine_content_vectors = []

    data = pd.read_csv('data/refine_data.csv')

    with open('data/index.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['raw_title', 'raw_content', 'refine_content', 'id'])

        for idx, row in data.iterrows():
            raw_title = row['title']
            id =  row['id']
            raw_content = row['content']
            refine_content = row['refine_content']

            raw_content_vector = len_safe_get_embedding(raw_content, average=True)
            refine_content_vector = len_safe_get_embedding(refine_content, average=True)

            if raw_content_vector is None or refine_content_vector is None:
                print('[continue] value is None')
                continue

            raw_content_vectors.append(raw_content_vector)
            refine_content_vectors.append(refine_content_vector)

            csvwriter.writerow([raw_title, raw_content, refine_content, id])

    np.save('data/raw_content_vector.npy', np.array(raw_content_vectors))
    np.save('data/refine_content_vector.npy', np.array(refine_content_vectors))

if __name__ == '__main__':
    data_conversion()

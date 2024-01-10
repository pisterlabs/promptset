import os
import numpy as np
from tqdm import tqdm
from vectordb_api import VectorDBAPI, flatten


def create_data():
    from text_embedder_processors import TextEmbedderProcessor, TextEmbeddingMethods
    from models.openai_model import openai_summary
    from models.bart import summarize_text_with_bart
    from models.gpt2 import summarize_text_gpt2
    from models.vertext_ai_model import vertext_ai_summary
    folder_path = "extracted-text"
    filenames = os.listdir(folder_path)
    st_embs = []
    vertex_embs = []
    tp_st = TextEmbedderProcessor("", method=TextEmbeddingMethods.ST)
    tp_vertex = TextEmbedderProcessor("", method=TextEmbeddingMethods.VERTEX)
    bart_summaries = []
    gpt2_summaries = []
    vertex_ai_summaries = []
    openai_summaries = []
    for i, filename in tqdm(enumerate(sorted(filenames))):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            text = f.read()

        openai_summaries.append(openai_summary(text[:min(len(text) - 1, 27_000)]))
        bart_summaries.append(summarize_text_with_bart(text))
        gpt2_summaries.append(summarize_text_gpt2(text))
        vertex_ai_summaries.append(vertext_ai_summary(text))

        tp_st.text = text
        tp_vertex.text = text

        st_embs.append(tp_st.process())
        vertex_embs.append(tp_vertex.process())

    np.save(file="data/summaries/bart_summaries.npy", allow_pickle=True,
            arr=np.array(bart_summaries))
    np.save(file="data/summaries/open_ai_summaries.npy", allow_pickle=True,
            arr=np.array(openai_summaries))
    np.save(file="data/summaries/vertex_ai_summaries.npy", allow_pickle=True,
            arr=np.array(vertex_ai_summaries))
    np.save(file="data/summaries/gpt2_summaries.npy", allow_pickle=True,
              arr=np.array(gpt2_summaries))

    np.save(file="data/embs/entire_st_embs.npy", allow_pickle=True, arr=np.array(
        st_embs))
    np.save(file="data/embs/entire_vertex_embs.npy", allow_pickle=True, arr=np.array(
        vertex_embs))


def load_data():
    folder_path = "extracted-text"
    filenames = os.listdir(folder_path)
    vector_db_api = VectorDBAPI(index_name="main", renew_index=True)
    vertex_embs = np.load(file="data/embs/entire_vertex_embs.npy", allow_pickle=True)

    for i, filename in tqdm(enumerate(sorted(filenames))):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            text = f.read()
        embedding = vertex_embs[i]
        final_content = filename + "---" + text
        vector_db_api.insert(text=final_content[:min(511, len(final_content))],
                             embedding=list(flatten(embedding)))


def main():
    create_data()
    load_data()


if __name__ == "__main__":
    main()

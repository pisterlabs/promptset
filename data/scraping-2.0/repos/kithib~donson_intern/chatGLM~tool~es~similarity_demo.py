"""
    pip install sentence-transformers
    pip install langchain
    pip install llama-index
"""
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from llama_index import LangchainEmbedding
from llama_index.embeddings import LangchainEmbedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
embed_model = LangchainEmbedding(embeddings)

# 512 dims
def get_similarity(text1,text2,local_embedding=False):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
    embed_model = LangchainEmbedding(embeddings)
    # text1 = "今天天气不错，出去玩"
    # text2 = "今天天气很好，不去玩了"
    t1 = embed_model.get_query_embedding(text1)
    t2 = embed_model.get_query_embedding(text2)
    res = embed_model.similarity(t1,t2)
    print(len(t1),t1)
    print(len(t2),t2)
    return res


def get_embedding(text):
    embedding = embed_model.get_query_embedding(text)
    return embedding


if __name__ == "__main__":

    # text1 = """
    # Dr. Gandara provides expert knowledge and award-winning patient interaction skills designed to present and interpret complex concepts such as molecular testing in language easily understood by patients and their care givers. He has been acknowledged by patient advocacy groups for this expertise, including presentations through the Addario Lung Cancer Foundation Living Room, a highly regarded patient interaction group. His shared decision-making paradigm with patients is a model used in physician educational programs.
    # """
    # text2 = """
    # Dr. Gandara possesses extensive expertise and accolades for his exceptional patient interactions and ability to effectively communicate intricate ideas, such as molecular testing, in a manner that patients and their caregivers can readily comprehend. Patient advocacy groups have recognized his proficiency in this regard, and he has been invited to speak at esteemed platforms like the Addario Lung Cancer Foundation Living Room, which is highly regarded for fostering patient engagement. Furthermore, his patient-centric approach, characterized by shared decision-making, serves as a blueprint in physician educational programs.
    # """  # chatgpt改写
    # # %%
    # print(get_similarity(text1, text2))

    t = """Apple,Phone,Good camera,Young people"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
    embed_model = LangchainEmbedding(embeddings)
    emb =embed_model.get_query_embedding(t)
    print(emb)
    t = """Apple,Fruits,Delicious,Kids"""
    emb =embed_model.get_query_embedding(t)
    print(emb)
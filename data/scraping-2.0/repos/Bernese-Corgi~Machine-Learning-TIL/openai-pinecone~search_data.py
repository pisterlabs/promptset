import os
import openai
import pandas as pd

from create_index import get_pinecone, load_data
from openai.embeddings_utils import get_embedding

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
EMBEDDING_MODEL = "text-embedding-ada-002"

def init_openai():
    openai.api_key = OPENAI_API_KEY
    
def query_post(*, query: str, namespace: str = 'content', top_k: int = 5):
    post_df = load_data('data/blog_posts.csv')
    content_mapped = dict(zip(post_df.vector_id, post_df.content_text))
    
    embedded_query = get_embedding(text=query, engine=EMBEDDING_MODEL)
    
    print(embedded_query)
    
    index = get_pinecone()

    print(index)

    query_result = index.query(
        vector=embedded_query,
        namespace=namespace,
        top_k=top_k
    )
    
    print(query_result)
    
    if not query_result.matches:
        print('no query result')

    matches = query_result.matches
    ids = [res.id for res in matches]
    scores = [res.score for res in matches]
    df = pd.DataFrame({
        'id': ids,
        'score': scores,
        'content': [content_mapped[_id] for _id in ids]
    })

    counter = 0

    for _, value in df.iterrows():
        counter += 1
        print(f"\npost id: {value.id}, score: {value.score}, content: {value.content}\n")

    print(f"total: {counter}")

    return df

if __name__ == "__main__":
    init_openai()
    query_post(query='맘스보드와 노리터보드 비교')
    # while True:
    #     user_input = input('질문해보세요 >> ')

    #     if user_input == ':q':
    #         break
    #     else:
    #         result = query_post(query=user_input)
            
    #         print(f"{result}\n\n")


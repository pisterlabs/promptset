import pandas as pd
from openai.embeddings_utils import get_embeddings, cosine_similarity, get_embedding
import openai, os, backoff

from llm_course.c09.gen_taobao_titles import generate_data_by_prompt

openai.api_key = os.environ.get("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"

batch_size = 100


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


# search through the reviews for a specific product
def search_product(df, query, n=3, pprint=True):
    product_embedding = get_embedding(
        query,
        engine=embedding_model
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .product_name
    )
    if pprint:
        for r in results:
            print(r)
    return results


if __name__ == '__main__':
    clothes_prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
    clothes_data = generate_data_by_prompt(clothes_prompt)
    clothes_product_names = clothes_data.strip().split('\n')
    clothes_df = pd.DataFrame({'product_name': clothes_product_names})
    clothes_df.product_name = clothes_df.product_name.apply(lambda x: x.split('.')[1].strip())
    clothes_df.head()


    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
        embeddings += batch_embeddings

    df["embedding"] = embeddings
    df.to_parquet("data/taobao_product_title.parquet", index=False)
    df = pd.concat([df, clothes_df], axis=0)
    df = df.reset_index(drop=True)
    display(df)

    results = search_product(df, "自然淡雅背包", n=3)

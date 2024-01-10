import openai, os, backoff
import pandas as pd
from openai.embeddings_utils import get_embeddings

COMPLETION_MODEL = "text-davinci-003"
embedding_model = "text-embedding-ada-002"

batch_size = 100

def generate_data_by_prompt(prompt):
    response = openai.Completion.create(
        engine = COMPLETION_MODEL,
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 2048,
        top_p = 1,
    )
    return response.choices[0].text

prompt = """请你生成100条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。"""
data = generate_data_by_prompt(prompt)

product_names = data.strip().split('\n')
df = pd.DataFrame({'product_name': product_names})
df.product_name = df.product_name.apply(lambda x: x.split('.')[1].strip())
# print(df.head())

clothes_prompt = """请你生成100条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
clothes_data = generate_data_by_prompt(clothes_prompt)
clothes_product_names = clothes_data.strip().split('\n')
clothes_df = pd.DataFrame({'product_name': clothes_product_names})
clothes_df.product_name = clothes_df.product_name.apply(lambda x: x.split('.')[1].strip())
clothes_df.head()

df = pd.concat([df, clothes_df], axis=0)
df = df.reset_index(drop=True)
print(df)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embedding_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0,len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
        return embeddings


prompts = df.product_name.tolist()
batch_size = 100
prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
embedding_model = "text-embedding-ada-002"
for batch in prompt_batches:
    batch_embeddings = get_embedding_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings
df["embedding"] = embeddings
df.to_parquet("/Users/zm/aigcData/my_taobao_produtct_title.parquet", index=False)


import openai
import os
import pandas as pd
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("sample/fortune_1000_revenue_2022.csv")
print(df.head())

def get_company_info(name: str, revenues: str, market_value: str, employees: str) -> str:
    context = f"{name} has {revenues} revenues, {market_value} market value and {employees} employees"
    return context

# DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)
df["info"] = df.apply(
                func=lambda df: get_company_info(
                            df['name'].strip(),
                            df['revenues'].strip(),
                            df['market_value'].strip(),
                            df['employees'].strip()),
                axis=1)
# print(df['info'])

# https://platform.openai.com/docs/guides/embeddings/second-generation-models
tokenizer = "cl100k_base"

# https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# total token
total_token = 0
for i, info in enumerate(df['info']):
    total_token += num_tokens_from_string(string=df['info'][i], encoding_name=tokenizer)
print(f"total_token: {total_token}")

# total price
total_price_est = total_token / 1000 * 0.0004
print(f"total_price_est: ${total_price_est}")

# https://platform.openai.com/docs/guides/embeddings/use-cases
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# text embedding
vector_result = get_embedding(text=df['info'][0])
print(f"text_embedding: {vector_result}")

# dimension
print(f"dimension: {len(vector_result)}")

# # text embedding for all the dataset
# df['ada_embedding'] = df['info'].apply(
#                         lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('embedded_fortune_1k_revenue.csv', index=False)
# print("embedding_completed...")

# """ output
# total_token: 24781
# total_price_est: $0.0099124
# embedding_completed...
# """


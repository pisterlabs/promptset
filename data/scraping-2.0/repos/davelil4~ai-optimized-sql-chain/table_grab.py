import pandas as pd
import os
from openai.embeddings_utils import get_embedding, cosine_similarity


# search through the reviews for a specific product
def search_table(user_query, top_n=3):
    
    df = pd.read_csv(os.path.join(
        os.getcwd(),
        'embeddings_database_gen/dataframe_csvs/schemata_embeddings_df.csv'
    ))
    
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    
    return res

def getTableContext(df: pd.DataFrame):
    return "".join(df['context'].tolist())

def getContext(question):
    return getTableContext(search_table(question)) + "---\n\n" + open(os.path.join(os.getcwd(), "gen.txt")).read()
    

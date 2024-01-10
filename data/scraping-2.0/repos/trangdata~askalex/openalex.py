import pyalex
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity

# from askalex import cosine_similarity, get_embedding


def find_abs(keywords, per_page=100):
    print("Finding pubs...")
    woi = pyalex.Works().search_filter(abstract=keywords).get(per_page=per_page)
    abs_df = pd.DataFrame(
        [
            {
                "title": e["title"],
                "abstract": e["abstract"],
                "url": e["doi"],
            }
            for e in woi
        ]
    )
    abs_df["abstract"] = abs_df["abstract"].apply(shorten_abstract)
    print("Done!")
    return abs_df


def shorten_abstract(text, max_words=500, max_length=300):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_length])
    else:
        return text


def get_embed(
    df,
    embedding_model="tcell_ada_embeddings",
    embedding_encoding="cl100k_base",  # this the encoding for text-embedding-ada-002
    max_tokens=8000,  # the maximum for text-embedding-ada-002 is 8191
    top_n=1000,
):
    print("Finding embeddings...")
    # omit reviews that are too long to embed
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.abstract.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)

    df["embedding"] = df.abstract.apply(
        lambda x: get_embedding(x, engine=embedding_model)
    )
    print("Done!")
    return df


def search_docs(
    df,
    user_query,
    embedding_model="tcell_ada_embeddings",
    top_n=10,
):
    # perform semantic search on these abstracts and find
    # the top 10 relevant abstracts
    embedding = get_embedding(user_query, engine=embedding_model)
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values("similarities", ascending=False).head(top_n)
    return res


def style_dataframe(df):
    # check that the input DataFrame has the correct columns
    expected_columns = ["similarities", "title", "abstract", "url"]
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in input DataFrame: {missing_columns}")

    styled_df = pd.DataFrame()
    styled_df["Publication"] = df.apply(
        lambda row: f'<p style="font-weight: bold; font-size: larger"><a href="{row["url"]}">{row["title"]}</a></p><p>{row["abstract"]}</p>',
        axis=1,
    )
    styled_df["Similarity"] = df["similarities"].apply(lambda x: f"{x:.3f}")

    return styled_df

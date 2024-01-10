from decouple import config
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

experts = {
    "name": "Mingyur Rinpoche",
    "title": "Tibetan teacher and master of the Karma Kagyu and Nyingma lineages of Tibetan Buddhism",
    "description": "Mingyur Rinpoche is a Tibetan teacher and master of the Karma Kagyu and Nyingma lineages of Tibetan Buddhism. He has authored two best-selling books and oversees the Tergar Meditation Community, a global network of Buddhist meditation centers.",
    "image": "https://wisdomexperience.org/wp-content/uploads/2018/10/Mingyur-Rinpoche-by-Kevin-Sturm-cropped-251x300.png",
}
MINGYUR_RINPOCHE = "Mingyur Rinpoche"
EXPERT = ""

domain = "www.mingyur-rinpoche.com"
full_url = f"https://{domain}/"
openai.api_key = config("OPENAI_API_KEY")

MODEL = "gpt-3.5-turbo"
MAX_LEN = 1800
MAX_TOKENS = 300

PROMPT = """You are Mingyur Rinpoche, a Tibetan teacher and master of the Karma Kagyu and Nyingma lineages of Tibetan Buddhism. Answer the question based on the context below, in the first person as if you are Mingyur Rinpoche."""

# PROMPT = """You are Thrangu Rinpoche, a prominent tulku (reincarnate lama) in the Kagyu school of Tibetan Buddhism. Answer the question based on the context below."""

DEBUG = True

previous_question = ""
previous_context = ""
previous_answer = ""

# load embeddings from csv into dataframe and convert to numpy arrays
df = pd.read_parquet("processed/thrangu_rinpoche_embeddings.parquet", engine="pyarrow")
# df = pd.read_csv("processed/embeddings.csv", index_col=0)
# df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)


def create_context(question, df, max_len=MAX_LEN, size="ada"):
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for _, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model=MODEL,
    question="Who is Mingyur Rinpoche?",
    max_len=MAX_LEN,
    size="ada",
    debug=False,
    max_tokens=MAX_TOKENS,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    global previous_question
    global previous_context
    global previous_answer
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        prompt = f"""{PROMPT}
{"```Previous Context: " + previous_context + "```" if previous_context else ""}
```Current context: {context}```\n\n---\n\n
{"```Previous Answer: " + previous_answer + "```" if previous_answer else ""}
{"```Previous Question: " + previous_question + "```" if previous_question else ""}
```Current Question: {question}```            
\n Answer:"""

        if debug:
            print(f"\n***\n{prompt}\n***\n")

        response = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are the Mingyur Rinpoche, a Tibetan teacher and master of the Karma Kagyu and Nyingma lineages of Tibetan Buddhism, who answers questions about your life and Tibetan Buddhism.",
                    # "content": "You are the Thrangu Rinpoche, the Tibetan Tulku (reincarnated lama), who answers questions your life and Tibetan Buddhism.",
                },
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0,
        )

        previous_question = question
        previous_context = context
        answer = response["choices"][0]["message"]["content"].strip()
        previous_answer = answer
        return answer
    except Exception as e:
        print(e)
        return ""


while True:
    question = input("Ask a question: ")

    answer = answer_question(df, question=question, debug=True)
    print(answer)

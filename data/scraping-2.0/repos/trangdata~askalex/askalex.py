import openai
import os
import numpy as np
from openai.embeddings_utils import get_embedding, distances_from_embeddings

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = os.getenv("OPENAI_PROXY")

# def get_embedding(text, engine="text-embedding-ada-002"):  # model = "deployment_name"
#     return client.embeddings.create(input=[text], model=engine).data[0].embedding
#     # return openai.Embedding.create(input=text, engine=engine)["data"][0]["embedding"]

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = get_embedding(question, engine="tcell_ada_embeddings")

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embedding"].values, distance_metric="cosine"
    )
    # df["sim"] = cosine_similarity(q_embeddings, df["embedding"].values)

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["abstract"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    question,
    df,
    engine="T-Cell-Phenotype",  # "GPT-4-32k",
    max_len=4097,
    size="ada",
    debug=False,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    if question is None:
        return ""

    template = (
        "You are an intelligent assistant helping users with their questions. "
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Answer the following question using only the data provided in the sources below. "
        + "For tabular information return it as an html table. Do not return markdown format. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. "
        + "\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer: "
    )

    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    prompt = template.format(context=context, question=question)
    try:
        return trim_incomplete_sentence(complete_model(prompt, engine, stop_sequence))
    except Exception as e:
        print(e)
        return ""


def trim_incomplete_sentence(paragraph):
    sentences = paragraph.split(". ")
    # if the last sentence is complete
    if sentences[-1].endswith("."):
        return paragraph
    # else, remove it
    trimmed_paragraph = ". ".join(sentences[:-1])
    trimmed_paragraph += "."
    return trimmed_paragraph


def complete_model(
    prompt,
    engine,
    stop_sequence,
):
    model = [engine]

    if "gpt" in model:
        max_tokens = 10000
    else:
        n_tokens = len(prompt) // 4
        max_tokens = 3880 - n_tokens

    if model == "gpt-4-32k" or model == "gpt-4":
        response = openai.ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
            engine=engine,
        )
        return response["choices"][0]["message"]["content"]
    else:
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            engine=engine,
        )
        return response["choices"][0]["text"]

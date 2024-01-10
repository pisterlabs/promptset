import random
import streamlit as st
import pickle
import dotenv
import numpy as np
from tqdm import tqdm
import openai
import os
import faiss
import pandas as pd
import streamlit_pandas as sp
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load concept_descriptions function
# load concept embedding for all API calls
st.write("Cold start - loading concept embeddings...")


@st.cache_data
def load_concept_descriptions():
    print("loading concept descs")
    concept_descriptions = pd.read_pickle(
        "/Users/danielgeorge/Documents/work/ml/bioconceptvec-explorer/bioconceptvec-explorer/mappings/concept_descriptions.pkl"
    )
    return concept_descriptions


# Load sentence_embeddings function


@st.cache_data
def load_sentence_embeddings():
    sentence_embeddings = np.load("./description_embeddings.npy")
    return sentence_embeddings


# Load sentences function


@st.cache_data
def load_sentences():
    with open("./sentences.txt") as f:
        sentences = f.readlines()
    return sentences


@st.cache_data
def load_concept_values():
    # load concept embedding for all API calls
    print("Cold start - loading concept embeddings...")
    with open("./embeddings/concept_glove.json") as json_file:
        concept_vectors = json.load(json_file)
        concept_keys = list(concept_vectors.keys())
        return np.array(list(concept_vectors.values()), dtype=np.float32)


@st.cache_data
def load_rev_concept_description():
    print("loading concept descriptions...")
    with open("./mappings/concept_descriptions.pkl", "rb") as f:
        concept_descriptions = pickle.load(f)
        rev_concept_descriptions = {}
        for key, value in tqdm(concept_descriptions.items()):
            if type(value) == list and len(value) == 0:
                continue
            elif type(value) == list and len(value) > 0:
                rev_concept_descriptions[value[0]] = key
            else:
                rev_concept_descriptions[value] = key


with open("./embeddings/concept_glove.json") as json_file:
    print("loading concept glove.json")
    concept_vectors = json.load(json_file)
    concept_keys = list(concept_vectors.keys())
    concept_values = np.array(list(concept_vectors.values()), dtype=np.float32)

# Load the necessary data
concept_descriptions = load_concept_descriptions()
sentence_embeddings = load_sentence_embeddings()
sentences = load_sentences()
concept_values = load_concept_values()
rev_concept_descriptions = load_concept_descriptions()

# Load the model
model = SentenceTransformer("bert-base-nli-mean-tokens")

# Initialize the index
d = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(sentence_embeddings)

st.write("Done!")


def load_openai_key(path):
    dotenv.load_dotenv(path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def get_prompt(query: str):
    return f"""
        What does this mean analogically? I found this by doing equations with vector embeddings.
        This is similar to how King - Man + Woman = Queen for word2vec. I'm trying to reason why this makes sense.

        {query}

        Really try to think outside the box to find why this could be reasonable. Use this as a generative way to help think of biological hypotheses.
        """


def gpt(prompt):
    load_openai_key("./.env")
    messageList = [
        {
            "role": "system",
            "content": "You are a helpful chatbot that helps people understand biology.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    feature_string = completion.choices[0].message.content

    return feature_string


print("Done!")


def compute_expression(
    expression: list,
    k: int = 10,
    useCosineSimilarity: bool = True,
) -> dict:
    # print(f"Computing expression: {expression}")

    if expression[0] != "-" and expression[0] != "+":
        expression = ["+", *expression]

    # split expression into groups of 2 (sign, concept)
    matches = [expression[i : i + 2] for i in range(0, len(expression), 2)]
    # compute x vector
    result = np.zeros(np.array(concept_values[0]).shape, dtype=np.float32)
    for match in matches:
        sign, variable = match
        # print(f"Variable: {variable} | Sign: {sign}")
        if sign == "-":
            result -= concept_vectors[variable]
        elif sign == "+":
            result += concept_vectors[variable]
        else:
            raise ValueError(f"Invalid operator: {sign}")

    similarities = None
    if useCosineSimilarity:
        # compute similarity between x vector and all other vectors
        similarities = cosine_similarity(concept_values, [result]).flatten()
    else:
        # compute distance between x vector and all other vectors
        similarities = np.linalg.norm(concept_values - result, axis=1).flatten()

    # get index of top k similarities
    top_k_indices = np.argpartition(similarities, -k)[-k:]

    # get top k most similar concepts as a dict
    top_concepts = {concept_keys[i]: float(similarities[i]) for i in top_k_indices}
    top_concepts = dict(
        sorted(top_concepts.items(), key=lambda item: item[1], reverse=True)
    )
    return top_concepts


def autosuggest(query: str, limit: int) -> list:
    # filter concept vectors based on whether query is a substring
    query = query.lower()
    descs = list(concept_descriptions.values())
    for i in range(len(descs)):
        if type(descs[i]) == list and len(descs[i]) > 0:
            descs[i] = descs[i][0]
        elif type(descs[i]) == list and len(descs[i]) == 0:
            descs[i] = ""

    descs = [i for i in descs if i is not None and i != ""]
    lower_concept_descs = map(lambda x: x.lower(), descs)
    result = [concept for concept in lower_concept_descs if query in concept]
    return result[:limit]


def get_similar_concepts(concept_query: str, k: int) -> list:
    # convert from concept description to concept id
    if ";" in concept_query:
        concept_query = concept_query.split(";")[0]
    concept_query = rev_concept_descriptions[concept_query]
    concept = concept_vectors[concept_query]
    similarities = cosine_similarity(concept_values, [concept]).flatten()
    top_concepts = {}
    for concept, similarity in zip(concept_vectors.keys(), similarities):
        top_concepts[concept] = similarity
    top_concepts = dict(
        sorted(top_concepts.items(), key=lambda item: item[1], reverse=True)[:k]
    )
    return top_concepts


def free_var_search(term: str, sim_threshold=0.7, n=100, top_k=3, use_gpt=False):
    print("Running free var search!!")
    term_vec = concept_vectors[term]
    expressions = []

    # randomly pick 1000 pairs of concepts for b, c
    concepts = list(concept_vectors.keys())
    equations = []
    for _ in range(n):
        b, c = random.sample(concepts, 2)
        equations.append([term, "+", b, "-", c])

    print("Solving equations...")
    good_equations = []
    for equation in tqdm(equations):
        concept, sim = compute_expression(
            equation,
            k=1,
        ).popitem()
        if sim > sim_threshold and concept not in equation:
            print(f"Equation: {equation} | Concept: {concept} | Similarity: {sim}")
            good_equations.append((equation, concept, sim))
            print(f"Expression: {equation} | Solution: {concept} | Similarity: {sim}")

    df = pd.DataFrame(good_equations, columns=["Equation", "Concept", "Similarity"])
    # Sort by similarity
    df = df.sort_values(by=["Similarity"], ascending=False)
    df = df.reset_index(drop=True)
    # Pick top k
    df = df.head(10)

    eq_mapped = []
    for row in good_equations:
        eq_mapped.append(
            " ".join(
                [str(concept_descriptions[i]) for i in row[0] if i != "+" and i != "-"]
            )
        )
    df["Equation_mapped"] = eq_mapped
    df["Concept Description"] = df["Concept"].apply(lambda x: concept_descriptions[x])

    # now we use gpt to generate a rationale for each equation using the prompt
    if use_gpt:
        rationales = []
        for row in tqdm(df.iterrows()):
            mapped_eq = row[1]["Equation_mapped"]
            prompt = get_prompt(mapped_eq)
            rationales.append(gpt(prompt))

        df["Rationale"] = rationales

    df.to_csv("results.csv", index=False)

    return df


def process_input(user_input):
    k = 8
    xq = model.encode([user_input])
    D, I = index.search(xq, k)
    options = [f"{i}: {sentences[i]}" for i in I[0]]
    return options


def select_option(options):
    selected_option = st.selectbox("Select a similar concept:", options)
    if selected_option:
        st.write("You selected:", selected_option)
    return selected_option


# Set up the Streamlit page
st.title("BioConceptVec Exploration App")

# Get the user's input
user_input = st.text_input("Enter a concept:")

if user_input:
    options = process_input(user_input)
    if options:
        option = select_option(options)
        if option:
            start_index = option.find(":") + 1
            end_index = option.find("|")
            extracted_string = option[start_index:end_index].strip()
            st.write(extracted_string)
            # Make an input box from 0.0 to 1.0 by increments of 0.1 multiselect
            threshold = st.multiselect(
                "Select a threshold:", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            )
            if threshold:
                threshold = threshold[0]
                free_var_search(extracted_string, threshold, use_gpt=True, top_k=10)
                import streamlit as st
                import pandas as pd

                # Load the CSV file
                data = pd.read_csv("results.csv")

                # Display a download button
                st.download_button(
                    label="Download CSV",
                    data=data.to_csv(),
                    file_name="res.csv",
                    mime="text/csv",
                )

                # Show the dataframe
                sp.write(data)

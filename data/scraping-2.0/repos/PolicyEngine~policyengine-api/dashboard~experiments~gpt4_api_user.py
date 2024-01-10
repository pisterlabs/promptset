import streamlit as st
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import json
import requests
import time
import os

# Point OpenAI to the API key
openai.api_key = os.environ["OPENAI_API_KEY"]


@st.cache_data
def get_embeddings_df():
    embeddings_df = pd.read_csv(
        "parameter_embeddings.csv.gz", compression="gzip"
    )
    embeddings_df.parameter_embedding = (
        embeddings_df.parameter_embedding.apply(lambda x: eval(x))
    )
    return embeddings_df


@st.cache_data
def embed(prompt, engine="text-embedding-ada-002"):
    return get_embedding(prompt, engine=engine)


embeddings_df = get_embeddings_df().copy()

st.title("GPT-4 PolicyEngine analysis")

# Text box for people to write "change the basic rate of income tax to 20%", single line

question = st.text_input(
    "Enter a policy question",
    "Revenue impact of changing the basic rate of income tax to 25%",
)

# First task: convert to:
# { 'hmrc.income_tax.rates.uk[0].rate': { '2023-01-01.2024-01-01': 0.25 } }

# We're going to do this with GPT4- we'll pass in the question, the related parameter metadata, and the task description.

# First, get related parameters via a search through GPT-embedded parameters.


# columns are name, json, parameter_embedding

# Get the top 5 most similar parameters

embedding = embed(question)
embeddings_df["similarities"] = embeddings_df.parameter_embedding.apply(
    lambda x: cosine_similarity(x, embedding)
)

top5 = (
    embeddings_df.sort_values("similarities", ascending=False)
    .head(5)["json"]
    .values
)
# display in streamlit


def write_prompt(question, relevant_parameters):
    prompt = f"""Question: {question}
    Here's some metadata about the parameters that are relevant to your question:
    Relevant parameters: {relevant_parameters}

    Task: Convert the question into a reform of the syntax below. If not specified, assume the reform starts on 2023-01-01 and ends on 2024-01-01.
    

    Reform syntax you should write in: {{ "parameter_name": {{ "start_date.end_date": value }} }}
    e.g. "basic rate at 25%" -> {{ "hmrc.income_tax.rates.uk[0].rate": {{ "2023-01-01.2024-01-01": 0.25 }} }}
    Reply only with valid JSON.
    """

    return prompt


reform_generation_prompt = write_prompt(question, top5)

# Now, pass the prompt to GPT-4


@st.cache_data
def get_gpt4_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )["choices"][0]["message"]["content"]
    return response


response = get_gpt4_response(reform_generation_prompt)

reform = json.loads(response)

# Next step: register the reform on the PolicyEngine API


@st.cache_data
def get_policyengine_impact(reform):
    payload = {"data": reform}
    url = "https://api.policyengine.org/uk/policy"

    response = requests.post(url, json=payload)

    # json.result.policy_id

    policy_id = response.json()["result"]["policy_id"]

    # Next step: run the reform on the PolicyEngine API

    url = f"https://api.policyengine.org/uk/economy/{policy_id}/over/1?time_period=2023&region=uk"

    response = requests.get(url).json()

    # if 'status' = 'computing', wait 5 seconds and try again until 'status' = 'ok'

    while response["status"] == "computing":
        time.sleep(5)
        response = requests.get(url).json()

    impact = response["result"]

    return impact


# Now, get GPT4 to answer the question with the impact JSON.

impact = get_policyengine_impact(reform)

question_answer_prompt = f"""Relevant economic impact simulation data: {impact}

    Question: {question}

    Task: Write a concise single-sentence response to the question, using only the impact data. If you can't from the data, say it.

    Give financial amounts in short-form (e.g. £1bn, 250m, etc.).
"""

answer = get_gpt4_response(question_answer_prompt)

st.write(answer)

st.write("## How I worked it out")

with st.expander("My prompt for understanding your reform"):
    st.write(reform_generation_prompt)

with st.expander("What I thought your reform was"):
    st.write(reform)

with st.expander("My prompt for answering your question"):
    st.write(question_answer_prompt)

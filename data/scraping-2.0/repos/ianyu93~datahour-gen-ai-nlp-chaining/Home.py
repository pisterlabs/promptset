import concurrent
import json

import openai
import pandas as pd
import streamlit as st

from src import queries
from src.utils import obj_to_json

# Load example data and documentation
with open("assets/examples.json") as f:
    EXAMPLES = json.load(f)

# Streamlit app title and overview
st.title("NLP Chaining Example")
st.markdown(
    """
LLMs are very powerful text generation models. Most NLP tasks can be framed as a text generation task via prompt
engineering. In this app, we will show how to chain NLP tasks with GenAI.

We create a chaining where:

1. **Natural Language Generation**: We will generate a list of named entity types given a persona and purpose.
3. **Natural Language Inference**: We will pass the extracted named entities to an NLI task to act as a proxy to data
quality check.
2. **Named Entity Recognition**: We will pass the generated list of named entity types to a NER model to extract the
named entities from a given article.
"""
)

st.markdown(
    """
    ### Setup OpenAI API Key
    First, we need to setup the OpenAI API key. You can find your API key at
    https://platform.openai.com/account/api-keys.
    """
)

openai.api_key = st.text_input("OpenAI API Key", type="password")

st.markdown(
    """
### Step 1: Natural Language Generation

Typically, users should already have specific named entities in their mind to create a Named Entity Recognition task.
But in some business cases, users may not already know what named entities they want to extract. In this case, we can
use a Natural Language Generation task to suggest a list of named entities.
"""
)

# User inputs for persona and purpose
PERSONA = st.text_input(
    "Persona",
    EXAMPLES["merchandiser"]["persona"],
    help="Enter a persona, e.g. merchandiser, teacher, writer.",
)
PURPOSE = st.text_area(
    "Purpose",
    EXAMPLES["merchandiser"]["purpose"],
    help="Enter a purpose, e.g. to sell more products, to teach students, to write a book.",
)

# Initialize session state variables if not already defined
for key in [
    "entity_types",
    "entity_types_to_display",
    "entities",
    "entities_to_display",
    "evaluations",
    "evaluations_to_display",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# Generate Entity Types button
if st.button("Generate Entity Types"):
    entity_types = queries.entity_type_suggestions(PERSONA, PURPOSE)
    st.session_state["entity_types"] = entity_types

    entity_types_to_display = list(
        map(lambda x: obj_to_json(x), entity_types.entity_types)
    )
    st.session_state["entity_types_to_display"] = entity_types_to_display

# Display Entity Types DataFrame
if st.session_state["entity_types_to_display"]:
    st.dataframe(st.session_state["entity_types_to_display"])

st.markdown(
    """
### Step 2: Named Entity Recognition
Once we have a list of named entities, we can pass them to a Named Entity Recognition task to extract the named
entities from a given article.
"""
)
# Text input for user-provided text
TEXT = st.text_area("Text", EXAMPLES["merchandiser"]["text"], height=500)

# Extract Entities button
if st.button("Extract Entities"):
    entity_types = st.session_state["entity_types"]
    entities = queries.ner(entity_types, TEXT)
    st.session_state["entities"] = entities
    entities_to_display = list(map(lambda x: obj_to_json(x), entities.entities))
    st.session_state["entities_to_display"] = entities_to_display

# Display extracted Entities DataFrame
if st.session_state["entities_to_display"]:
    st.dataframe(st.session_state["entities_to_display"])

st.markdown(
    """
### Step 3: Natural Language Inference
Once we have the extracted named entities, we can pass them to a Natural Language Inference task to act as a proxy to
data quality check. Note that this type of automatic check is only for proxy, not a replacement for human review.
Typically, you would treat human evaluation as golden standard, and use the model to flag potential issues for human
review.
"""
)
# NLI Data Check button
if st.button("NLI Data Check"):
    entities = st.session_state["entities"]

    def nli_check(entity, text):
        output = queries.ner_data_quality_check(entity, text)
        output = obj_to_json(output)
        output["entity_type"] = entity.label
        output["entity_text"] = entity.text
        return output

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        evaluations = list(
            executor.map(lambda x: nli_check(x, TEXT), entities.entities)
        )
    st.session_state["evaluations"] = evaluations

# Display NLI Data Check results
if st.session_state["evaluations"]:

    def cooling_highlight(val):
        if val == "entailment":
            color = "#90EE90"
        elif val == "#F5F5DC":
            color = "#FFC0CB"
        else:
            color = "red"
        return f"background-color: {color}"

    df = pd.DataFrame(st.session_state["evaluations"]).style.applymap(
        cooling_highlight, subset=["label"]
    )
    st.dataframe(
        df, column_order=["entity_type", "entity_text", "label", "explanation"]
    )

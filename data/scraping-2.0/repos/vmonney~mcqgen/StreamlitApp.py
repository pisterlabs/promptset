"""Streamlit app for MCQs Creator using LangChain."""

import json
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain.callbacks import get_openai_callback

from src.mcqgenerator.logger import logging
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.utils import get_table_data, read_file

# loading json file
RESPONSE_JSON = json.loads(
    Path(
        "/Users/valentinmonney/Desktop/Generative_ai/mcqgen/Response.json",
    ).read_text(),
)

# Retrieve the configured logger
logger = logging.getLogger()

# Create a title
st.title("MCQs Creator App with LangChain 🦜⛓️")

# Create a form using st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or a TXT file", type=["pdf", "txt"])

    # Input fields
    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50, value=10)

    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz difficulty
    difficulty_levels = ["Easy", "Moderate", "Hard", "Expert"]
    difficulty_level = st.selectbox(
        "Select Quiz Difficulty Level",
        options=difficulty_levels,
    )

    # Add Button
    button = st.form_submit_button(label="Generate MCQs")

    # Check if the button is clicked and all fields are filled
    if (
        button
        and uploaded_file is not None
        and mcq_count
        and subject
        and difficulty_level
    ):
        with st.spinner("Loading..."):
            try:
                text = read_file(uploaded_file)
                # Count tokens and the cost of API calls
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "difficulty_level": difficulty_level,
                            "response_json": json.dumps(RESPONSE_JSON),
                        },
                    )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                logger.info(f"Total Tokens: {cb.total_tokens}")
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
                logger.info(f"Total Cost: {cb.total_cost}")
                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            quiz_df = pd.DataFrame(table_data)
                            quiz_df.index = quiz_df.index + 1
                            st.table(quiz_df)
                            # Display the review in a text box as well
                            st.text_area("Review", value=response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)

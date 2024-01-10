import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import random
import time

LOG = "questions.log"

@st.cache_data()
def load_data(file):
    """
    Load the data.
    """
    df = pd.read_csv(file, encoding="utf-8", delimiter=",")
    return pre_process(df)

def add_to_log(question):
    """
    Log the question
    """
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " ")
        f.write(question + "\n") 
        f.flush()

def pre_process(df):
    """
    Pre-process the data.
    """
    # Drop columns that start with "Unnamed"
    for col in df.columns:
        if col.startswith("Unnamed"):
            df = df.drop(col, axis=1)
    return df

def ask_question(question, system="You are a data scientist."):
    """
    Ask a question and return the answer.
    """ 
    client = OpenAI()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
        ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        stop = ["plt.show()", "st.pyplot(fig)"]
        )
    answer = response.choices[0].message.content
    return answer

def ask_question_with_retry(question, system="You are a data scientist.", retries=1):
    """
    Wrapper around ask_question that retries if it fails.
    Proactively wait for the rate limit to reset. Eg for a rate limit of 20 calls per minutes, wait for at least 2 seconds
    Compute delay using an exponential backoff, so we don't exceed the rate limit.
    """
    delay = 2 * (1 + random.random())
    time.sleep(delay)
    for i in range(retries):
        try:
            return ask_question(question, system=system)
        except Exception as e:
            delay = 2 * delay
            time.sleep(delay)
    return None

def prepare_question(description, question, initial_code):
    """
    Prepare a question for the chatbot.
    """
    return f"""
Context:
{description}
Question: {question}
Answer:
{initial_code}
"""

def describe_dataframe(df):
    """
    Describe the dataframe.
    """
    description = []
    # List the columns of the dataframe
    description.append(f"The dataframe df has the following columns: {', '.join(df.columns)}.")
    try:
        # For each column with a categorical variable, list the unique values
        if cols := check_categorical_variables(df):
            return f"ERROR: All values in a categorical variable must be strings: {', '.join(cols)}." 
        for column in df.columns:
            if df[column].dtype == "object" and len(df[column].unique()) < 10:
                description.append(f"Column {column} has the following levels: {', '.join(df[column].dropna().unique())}.")
            elif df[column].dtype == "int64" or df[column].dtype == "float64":
                description.append(f"Column {column} is a numerical variable.")
        description.append("Add a title to the plot.")
        description.append("Label the x and y axes of the plot.")
        description.append("Do not generate a new dataframe.")
    except Exception as e:
        add_to_log("Error: Unexpected error with the dataset.")
        return "Unexpected error with the dataset."
    return "\n".join(description)

def check_categorical_variables(df):
    """
    Check that all values of categorical variables are strings.
    """
    # Return [] if all values of categorical variables are strings
    # Return columns if not all values of categorical variables are strings
    return [column for column in df.columns if df[column].dtype == "object" 
        and not all(isinstance(x, str) for x in df[column].dropna().unique())]

def list_non_categorical_values(df, column):
    """
    List the non-categorical values in a column.
    """
    return [x for x in df[column].unique() if not isinstance(x, str)]

def code_prefix():
    """
    Code to prefix to the visualization code.
    """
    return """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 2.4))
"""
    
def generate_placeholder_question(df):
    return "Show the relationship between x and y."

def test_ask_question():
    system = "Write Python code to answer the following question. Do not include comments."
    question = "Generate a function that returns the ratio of two subsequent Fibonacci numbers."
    answer = ask_question_with_retry(question, system=system)
    print(answer)

def test_describe_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    description = describe_dataframe(df)
    print(description)

def test_visualize_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    question = "Show the relationship between a and c."
    description = describe_dataframe(df)
    initial_code = code_prefix()
    print(prepare_question(description, question, initial_code))

def test_visualize_dataframe_with_chat():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    question = "Show the relationship between a and c."
    description = describe_dataframe(df)
    initial_code = code_prefix()
    answer = ask_question_with_retry(prepare_question(description, question, initial_code))
    print(initial_code + answer)

st.title("Chat with your data")

uploaded_file = st.sidebar.file_uploader("Upload a dataset", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    with st.chat_message("assistant"):
        st.markdown("Here is a table with the data:")
        st.dataframe(df, height=200)

    question = st.chat_input(placeholder=generate_placeholder_question(df))

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        add_to_log(f"Question: {question}")
            
        description = describe_dataframe(df)
        if "ERROR" in description:
            with st.chat_message("assistant"):
                st.markdown(description)
        else:
            initial_code = code_prefix()
            with st.spinner("Thinking..."):
                answer = ask_question_with_retry(prepare_question(description, question, initial_code))
            with st.chat_message("assistant"):
                if answer:
                    script = initial_code + answer + "st.pyplot(fig)"
                    try:
                        exec(script)
                        st.markdown("Here is the code used to create the plot:")
                        st.code(script, language="python")
                    except Exception as e:
                        add_to_log("Error: Could not generate code to answer this question.")
                        st.info("I could not generate code to answer this question. " +
                            "Try asking it in a different way.")
                else:
                    add_to_log("Error: Request timed out.")
                    st.markdown("Request timed out. Please wait and resubmit your question.")
else:
    with st.chat_message("assistant"):
        st.markdown("Upload a dataset to get started.")

# if __name__ == "__main__":    
#     # test_ask_question()
#     # test_describe_dataframe()
#     # test_visualize_dataframe()
#     test_visualize_dataframe_with_chat()
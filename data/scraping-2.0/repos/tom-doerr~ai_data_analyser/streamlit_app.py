import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openai
import time
import streamlit.components.v1 as components
from PIL import Image

image = Image.open("res/logo.jpeg")

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.set_page_config(
    page_title="AI Data Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="res/logo.jpeg",
)

import leafmap.foliumap as leafmap
import requests
import datetime as dt
import numpy as np


if True:
    col1, col2, col3 = st.columns([12, 10, 10])
    with col1:
        st.write("")

    with col2:
        st.image("res/logo.jpeg", width=100)

    with col3:
        st.write("")


st.sidebar.title("Options")
num_successes_to_generate = st.sidebar.slider(
    "Number of Successes to Generate", 1, 500, 100
)
show_prompt_sent_to_codex = st.sidebar.checkbox(
    "Show prompt sent to codex", value=False
)

result = None


def get_first_x_lines(str_, num_lines):
    lines = str_.split("\n")[:num_lines]
    return "\n".join(lines)


def clear_screen_and_display_generated_readme(
    response, using_streamlit=False, code_box=None, text_mode=False
):
    if not code_box:
        code_box = st.empty()
    completion_all = ""
    key_str = str(time.time())
    while True:
        next_response = next(response)
        completion = next_response["choices"][0]["text"]
        completion_all = completion_all + completion
        if using_streamlit:
            if text_mode:
                code_box.markdown(completion_all)
            else:
                code_box.code(completion_all, language="python")
        if next_response["choices"][0]["finish_reason"] != None:
            break

    return completion_all


def get_csv_content():
    csv_file = st.file_uploader("Choose a CSV file", type="csv")

    if csv_file is None:
        st.write("No CSV file selected, using example data:")
        csv_file_name = "sample_csv.csv"
        with open(csv_file_name, "r") as f:
            csv_file_content = f.read()
    else:
        csv_file_content = csv_file.read().decode("utf-8")

    if csv_file_content is not None:
        df = pd.read_csv(io.StringIO(csv_file_content))
        st.table(df.head(100))
    else:
        st.text("Please upload a CSV file")
        st.stop()

    return csv_file_content


def ai_analytics():
    openai.api_key = st.secrets["openai_api_key"]

    csv_file_head = get_first_x_lines(csv_file_content, 10)

    user_command = st.text_input("Enter your command:")

    if user_command == "":
        st.text("Please enter a command")
        return

    instruction = "#!/usr/bin/env python3\n\n"
    instruction += "'''This script visualizes the content of the following csv file inside streamlit:\n\n"
    instruction += csv_file_head
    instruction += "'''\n\n"
    instruction += "import streamlit as st\n"
    instruction += "import pandas as pd\n"
    instruction += "import io\n"
    instruction += "import plotly.graph_objects as go\n"
    instruction += "import matplotlib.pyplot as plt\n\n"

    instruction += 'csv_file_content = st.file_uploader("Choose a CSV file", type="csv").read().decode("utf-8")\n\n'
    instruction += "# " + user_command + "\n"

    if show_prompt_sent_to_codex:
        st.write("Prompt sent to Codex:")
        st.code(instruction, language="python")

    print("instruction:", instruction)

    temperature = 0.7
    num_tokens = 200
    STOP_TOKENS = "\n\n"

    num_successes = 0
    num_attempts = 0
    times_rate_limit_excceeded = 0

    while True:
        code_box = st.empty()
        error_box = st.empty()
        while True:
            time.sleep(0.1)
            try:
                response = openai.Completion.create(
                    engine="code-davinci-002",
                    prompt=instruction,
                    temperature=temperature,
                    max_tokens=num_tokens,
                    stream=True,
                    stop=STOP_TOKENS,
                )

                completion_all = clear_screen_and_display_generated_readme(
                    response, using_streamlit=True, code_box=code_box
                )

                times_rate_limit_excceeded = 0
                exec(completion_all)
                error_box.empty()
                break
            except openai.error.RateLimitError:
                error_box.info("Rate limit exceeded, waiting a bit")
                time.sleep(2 ** times_rate_limit_excceeded)
                times_rate_limit_excceeded += 1
                error_box.empty()
            except Exception as e:
                print("e:", e)
                print("        type(e):", type(e))
                error_box.info(e)

            num_attempts += 1
            if num_attempts > 1000:
                st.warning("Too many attempts, stopping")
                return

        num_successes += 1
        if num_successes > num_successes_to_generate:
            st.write(f"More than {num_successes_to_generate} successes, exiting")
            return


col22, col23 = st.columns([3, 1])


with col23:
    csv_file_content = get_csv_content()

with col22:
    ai_analytics()

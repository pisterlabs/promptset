import json
import os
import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, TypedDict

import openai
import pandas as pd
import streamlit as st
from vega_datasets import data as vega_data

openai.api_key = os.environ.get("OPENAI_API_KEY")


### Utilities:


def name_to_variable(name: str) -> str:
    """Converts a name to a valid snake_case variable name."""
    name = re.compile(r"\s+").sub(" ", name).strip().replace(" ", "_").lower()
    # Remove all non-alphanumeric characters
    name = re.sub(r"[^A-Za-z0-9_]+", "", name)
    return name


def load_code_as_module(code: str) -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_loader("loaded_module", loader=None)
    assert spec is not None
    loaded_module = importlib.util.module_from_spec(spec)
    exec(code, loaded_module.__dict__)
    return loaded_module


def extract_first_code_block(markdown_text: str) -> str:
    if (
        "```" not in markdown_text
        and "return" in markdown_text
        and "def" in markdown_text
    ):
        # Assume that everything is code
        return markdown_text

    pattern = r"(?<=```).+?(?=```)"

    # Use re.DOTALL flag to make '.' match any character, including newlines
    first_code_block = re.search(pattern, markdown_text, flags=re.DOTALL)

    code = first_code_block.group(0) if first_code_block else ""
    code = code.strip().lstrip("python").strip()
    return code


def get_df_types_info(df: pd.DataFrame = None) -> str:
    column_info = {
        "dtype": [str(dtype) for dtype in df.dtypes],
        "inferred dtype": [
            pd.api.types.infer_dtype(column) for _, column in df.items()
        ],
        "missing values": df.isna().sum().to_list(),
    }

    return pd.DataFrame(
        column_info,
        index=df.columns,
    ).to_csv()


def get_column_descriptions_info(column_desc_df: pd.DataFrame) -> str:
    return "Column descriptions: \n\n" + column_desc_df.to_csv()


def get_df_stats_info(df: pd.DataFrame) -> str:
    return df.describe().to_csv()


def get_df_sample_info(df: pd.DataFrame, sample_size: int = 20) -> str:
    return df.sample(min(len(df), sample_size), random_state=1).to_csv()


class DataDescription(TypedDict):
    data_description: str
    columns: List[Dict[str, str]]
    chart_ideas: List[str]


def get_dataset_description_prompt(
    dataset_df: pd.DataFrame,
    column_desc_df: Optional[pd.DataFrame] = None,
    light: bool = False,
) -> str:
    prompt = f"""
I have a dataset (as Pandas DataFrame) that contains data with the following characteristics:

Column data types:

{get_df_types_info(dataset_df)}

{get_column_descriptions_info(column_desc_df) if column_desc_df is not None else ""}

Data sample:

{get_df_sample_info(dataset_df, sample_size=10 if light else 20)}
    """

    if light:
        return prompt

    return (
        prompt
        + f"""
Data statistics:

{get_df_stats_info(dataset_df)}
    """
    )


@st.cache_data(show_spinner=False)
def get_data_description(
    dataset_df: pd.DataFrame,
    openai_model: str = "gpt-3.5-turbo",
) -> DataDescription:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df)}

Please provide the following information based on this JSON template:

{{
    "data_description": "{{A concise description of the dataset}}",
    "columns": [
        {{ "name": "{{column name}}", "description": "{{A short description about the column}}" }},
    ]
    "chart_ideas": [
        "{{Concise description of your best chart or visualization idea}}",
        "{{Concise description of your second best chart or visualization idea}}",
        "{{Concise description of the third best chart or visualization idea}}",
        "{{Concise description of the fourth best chart or visualization idea}}",
    ]
}}

Please only respond with a valid JSON and fill out all {{placeholders}}:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps with describing data. The user will share some information about a Pandas dataframe, and you will create a description of the data based on a user-provided JSON template. Please provide your answer as valid JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]

    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response_json = json.loads(completion.choices[0].message.content)
    return response_json


@st.cache_data(show_spinner=False)
def generate_code(
    dataset_df: pd.DataFrame,
    task_instruction: str,
    method_name: str,
    description: str,
    openai_model: str = "gpt-3.5-turbo",
    column_desc_df: Optional[pd.DataFrame] = None,
) -> str:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df)}

Based on this data, please create a function that performs the following Data Visualization task with Plotly:

{task_instruction}

Please use the following template and implement all {{placeholders}}:

```
import pandas as pd
import numpy as np
from typing import *
import plotly

def {method_name}(data: pd.DataFrame) -> Union["plotly.graph_objs.Figure", "plotly.graph_objs.Data"]:
    \"\"\"
    {description}

    Visualize the data using the Plotly library.

    Args:
        data (pd.DataFrame): The data to visualize.

    Returns:
        Union[plotly.graph_objs.Figure, plotly.graph_objs.Data]: The plotly visualization figure object.
    \"\"\"
    # All task-specific imports here:
    import plotly

    {{Task implementation}}
```

Implement all {{placeholders}}, make sure that the Python code is valid. Please put the code in a markdown code block (```) and ONLY respond with the code:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps create python functions to perform data exploration and visualization tasks on a Pandas dataframe. The user will provide a description about the dataframe as well as a code template and instructions. Please only answer with valid Python code.",
        },
        {"role": "user", "content": user_prompt},
    ]

    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response = completion.choices[0].message.content.strip()

    code_block = extract_first_code_block(response)
    if not code_block and response:
        st.info(response, icon="🤖")
    return code_block


class TaskSummary(TypedDict):
    name: str
    method_name: str
    task_description: str
    emoji: str


@st.cache_data(show_spinner=False)
def get_task_summary(
    dataset_df: pd.DataFrame,
    task_instruction: str,
    openai_model: str = "gpt-3.5-turbo",
    column_desc_df: Optional[pd.DataFrame] = None,
) -> TaskSummary:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df)}

Based on this data, I have the following task instruction:

{task_instruction}

Please provide me the following information based on the task instruction:

{{
    "name": "{{Short Task Name}}",
    "method_name": "{{python_method_name}}",
    "task_description": "{{a concise description of the task}}",
    "emoji": "{{a descriptive emoji}}",
}}

The JSON needs to be compatible with the following TypeDict:

```python
class TaskSummary(TypedDict):
    name: str
    method_name: str
    task_description: str
    emoji: str
```

Please only respond with a valid JSON:
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. The user will share some characteristics of a dataset and a data exploration or visualization task instructions that the user likes to perform on this dataset. You will create some descriptive information about the task instruction based on a provided JSON template. Please only respond with a valid JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]
    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response_json = json.loads(completion.choices[0].message.content)
    return response_json


@dataclass
class ChartingRequest:
    instruction: Optional[str] = None
    generated_code: Optional[str] = None
    method_name: Optional[str] = None
    chart_obj: Optional[Any] = None
    exception: Optional[str] = None


@dataclass
class DatasetState:
    dataset: Optional[pd.DataFrame] = None
    data_description: Optional[DataDescription] = None
    column_descriptions: Optional[List[str]] = None
    history: List[ChartingRequest] = field(default_factory=list)


if "dataset" not in st.session_state:
    st.session_state["dataset"] = DatasetState()

charting_state: DatasetState = st.session_state["dataset"]


def clear_history() -> None:
    st.session_state["dataset"].history = []


def clear_dataset() -> DatasetState:
    st.session_state["dataset"] = DatasetState()
    return st.session_state["dataset"]


st.title("📈 ChartGPT")

with st.chat_message("assistant"):
    st.markdown(
        "Hi 👋 I'm ChartGPT. I can help you create charts from your data. To get started, please load your dataset below:"
    )

    with st.form(key="load_data"):
        vega_datasets = [
            dataset_name
            for dataset_name in vega_data.list_datasets()
            if dataset_name not in ["7zip", "annual-precip"]
        ]
        load_data: Optional[Callable] = None
        toy_dataset = st.selectbox(label="Select a Toy Dataset", options=vega_datasets)
        if toy_dataset is not None:
            dataset = getattr(vega_data, toy_dataset.replace("-", "_"))
            load_data = dataset
        if st.form_submit_button("📂 Load Data"):
            charting_state = clear_dataset()
            with st.spinner("📥 Loading data..."):
                charting_state.dataset = load_data() if load_data else None

if charting_state.dataset is None:
    st.stop()

with st.chat_message("assistant"):
    st.markdown("Thanks, I was able to load your data 🎉 Here is an overview:")

    st.dataframe(
        charting_state.dataset.head(min(len(charting_state.dataset), 5000)),
        use_container_width=True,
    )

with st.chat_message("assistant"):
    with st.spinner("🔬 Analyzing your data..."):
        if charting_state.data_description is None:
            charting_state.data_description = get_data_description(
                charting_state.dataset
            )
    st.markdown(charting_state.data_description["data_description"])
    st.markdown("Here is my interpretation of every column:")
    charting_state.column_descriptions = st.data_editor(
        pd.DataFrame(charting_state.data_description["columns"]),
        column_config={
            "name": "Column",
            "description": "Description",
        },
        use_container_width=True,
        disabled=["name"],
        hide_index=True,
    )
    st.markdown("Here is my favorite visualization idea:")
    preset_prompt = None
    chart_idea = charting_state.data_description["chart_ideas"][0].strip()
    if st.button("💡 " + chart_idea, help=chart_idea, use_container_width=True):
        clear_history()
        preset_prompt = chart_idea


def render_charting_response(chart_request: ChartingRequest) -> None:
    st.expander("Generated code").code(chart_request.generated_code, language="python")
    if chart_request.chart_obj:
        st.plotly_chart(chart_request.chart_obj)
    if chart_request.exception:
        st.error(chart_request.exception)


# Render chart history:
for chart_request in charting_state.history:
    st.chat_message("user").write(chart_request.instruction)
    with st.chat_message("assistant"):
        render_charting_response(chart_request)

# React to new requests:
prompt = st.chat_input("What do you like to visualize?") or preset_prompt
if prompt:
    st.chat_message("user").write(prompt)

    new_chart = ChartingRequest()
    new_chart.instruction = prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyzing instruction..."):
            task_summary = get_task_summary(
                charting_state.dataset,
                prompt,
                column_desc_df=charting_state.column_descriptions,
            )
            name = task_summary["emoji"] + " " + task_summary["name"]
            new_chart.method_name = name_to_variable(task_summary["method_name"])

            with st.spinner(f"Generating code for: {name}"):
                new_chart.generated_code = generate_code(
                    charting_state.dataset,
                    task_instruction=prompt,
                    method_name=new_chart.method_name,
                    description=task_summary["task_description"],
                    column_desc_df=charting_state.column_descriptions,
                )
            with st.spinner(f"Creating chart for: {name}"):
                assert new_chart.method_name is not None
                assert new_chart.generated_code is not None
                try:
                    loaded_module = load_code_as_module(new_chart.generated_code)
                    task_method = getattr(loaded_module, new_chart.method_name)
                    new_chart.chart_obj = task_method(data=charting_state.dataset)
                except Exception as ex:
                    exception_text = str(ex)
                    new_chart.exception = exception_text
                charting_state.history.append(new_chart)
                render_charting_response(new_chart)

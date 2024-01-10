import os
from dotenv import load_dotenv
import pandas as pd
import io
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.tools import PythonAstREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
from src.types import TableMapping
from src.prompt import (
    DATA_SCIENTIST_PROMPT_STR,
    SPEC_WRITER_PROMPT_STR,
    ENGINEER_PROMPT_STR,
)

load_dotenv()

if os.environ.get("DEBUG") == "true":
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = "llm-data-mapper"


NUM_ROWS_TO_RETURN = 5
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "synthetic")

# TODO: consider different models for different prompts, e.g. natural language prompt might be better with higher temperature
BASE_MODEL = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
)


def _get_data_str_from_df_for_prompt(df, num_rows_to_return=NUM_ROWS_TO_RETURN):
    return f"<df>\n{df.head(num_rows_to_return).to_markdown()}\n</df>"


def get_table_mapping(source_df, template_df):
    """Use PydanticOutputParser to parse the output of the Data Scientist prompt into a TableMapping object."""
    table_mapping_parser = PydanticOutputParser(pydantic_object=TableMapping)
    analyst_prompt = ChatPromptTemplate.from_template(
        template=DATA_SCIENTIST_PROMPT_STR,
        partial_variables={
            "format_instructions": table_mapping_parser.get_format_instructions()
        },
    )
    mapping_chain = analyst_prompt | BASE_MODEL | table_mapping_parser
    table_mapping: TableMapping = mapping_chain.invoke(
        {
            "source_1_csv_str": _get_data_str_from_df_for_prompt(source_df),
            "target_csv_str": _get_data_str_from_df_for_prompt(template_df),
        }
    )
    return pd.DataFrame(table_mapping.dict()["table_mappings"])


def _sanitize_python_output(text: str):
    """Remove markdown from python code, as prompt returns it."""
    _, after = text.split("```python")
    return after.split("```")[0]


def generate_mapping_code(table_mapping_df) -> str:
    """Chain two prompts together to generate python code from a table mapping: 1. technical spec writer, 2. python engineer"""
    writer_prompt = ChatPromptTemplate.from_template(SPEC_WRITER_PROMPT_STR)
    engineer_prompt = ChatPromptTemplate.from_template(ENGINEER_PROMPT_STR)

    writer_chain = writer_prompt | BASE_MODEL | StrOutputParser()
    engineer_chain = (
        {"spec_str": writer_chain}
        | engineer_prompt
        | BASE_MODEL
        | StrOutputParser()
        | _sanitize_python_output
    )
    return engineer_chain.invoke({"table_mapping": str(table_mapping_df.to_dict())})


def process_csv_text(value):
    """Process a CSV file into a dataframe, either from a string path or a file."""
    if isinstance(value, str):
        df = pd.read_csv(value)
    else:
        df = pd.read_csv(value.name)
    return df


def transform_source(source_df, code_text: str):
    """Use PythonAstREPLTool to transform a source dataframe using python code."""
    return PythonAstREPLTool(locals={"source_df": source_df}).run(code_text)

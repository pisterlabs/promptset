from typing import List

from pydantic import BaseModel, Field

from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser


class ColumnObjectiveEvaluation(BaseModel):
    reasons_for: List[str] = Field(
        description="List of reasons why the column meets the objective. "
        "This can be empty if there arent any good ones."
    )
    reasons_against: List[str] = Field(
        description="List of reasons why the column does not meet the objective. "
        "This can be empty if there arent any good ones."
    )
    score: float = Field(
        description="Likelihood that the column will assist in meeting the objective. "
        "A score between 0 and 1. "
        "0.0 means that the table wont be usefull at all. "
        "0.5 means that we dont know if the table will be usefull or not. "
        "1.0 means that the table is definitely usefull. "
    )


OBJ_EVAL_COL_DESC_TEMPLATE = (
    "You will be presented with an objective, tablename, column, the table schema and "
    "a table extract (table info), some additional entries from the column specified, "
    "and a guessed description of what the column contains."
    "\n\nTablename: {table_name}"
    "\n\nColumn: {column_name}"
    "\n\nTable Info (only a small sample of the data): {table_info}"
    "\n\nColumn extract (only a small sample of the data): {column_extract}"
    "\n\nColumn potential description: {infered_columns}"
    "\n\nYou are determine whether the column is useful in answering the objective; "
    "could the column be useful (when used in a sql query) to answer the objective?"
    "\n\nObjective: {objective}"
    "\n\nYou are only evaluating the {column_name} columns usefullness."
    "\n\n{format_instructions}"
)

OBJ_EVAL_COL_DESC_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=ColumnObjectiveEvaluation
)

OBJ_EVAL_COL_DESC_PROMPT = PromptTemplate(
    template=OBJ_EVAL_COL_DESC_TEMPLATE,
    input_variables=[
        "objective",
        "table_name",
        "column_name",
        "table_info",
        "column_extract",
        "infered_columns",
    ],
    partial_variables={
        "format_instructions": OBJ_EVAL_COL_DESC_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=OBJ_EVAL_COL_DESC_OUTPUT_PARSER,
)


def create_objective_evaluate_column_chain(**kwargs):
    return LLMChain(
        prompt=OBJ_EVAL_COL_DESC_PROMPT,
        **kwargs,
        output_parser=OBJ_EVAL_COL_DESC_OUTPUT_PARSER
    )

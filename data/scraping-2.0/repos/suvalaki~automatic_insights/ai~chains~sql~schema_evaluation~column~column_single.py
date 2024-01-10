from typing import List, Dict, Optional, Any, Tuple, Union

import pandas as pd

from pydantic import BaseModel, Field, root_validator
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.sql_database import SQLDatabase
from langchain.llms.base import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ai.chains.parallel_chain import ExtractChain, ParallelChain
from ai.utilities.sql_database import get_table


class ColumneEvaluatedDescription(BaseModel):
    column: str = Field(description="The column name being described")
    possible_description: str = Field(
        description="A detailed description and discussion of the column"
    )
    short_description: str = Field(
        description="A short summary description of the column and its contents"
    )


EVAL_COL_DESC_TEMPLATE = (
    "You will be presented with a tablename, column, the table schema and "
    "a table extract (table info), and some additional entries from the column specified. "
    "You are to evaulate what the column contains into detailed and short "
    "summaries."
    "The detailed summary should be exploratory and seeks to make inferences "
    "about the data present in the column. "
    "The short description should be a summary of only the important details "
    "previously provided. "
    "\n\nTablename: {table_name}"
    "\n\nColumn: {column_name}"
    "\n\nTable Info (only a small sample of the data): {table_info}"
    "\n\nColumn extract (only a small sample of the data): {column_extract}"
    "\n\n{format_instructions}"
)

EVAL_COL_DESC_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=ColumneEvaluatedDescription
)

EVAL_COL_DESC_PROMPT = PromptTemplate(
    template=EVAL_COL_DESC_TEMPLATE,
    input_variables=["table_name", "column_name", "table_info", "column_extract"],
    partial_variables={
        "format_instructions": EVAL_COL_DESC_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=EVAL_COL_DESC_OUTPUT_PARSER,
)


def get_column_samples(table: str, column: str, samples: int = 10):
    query = select(table).limit(samples)


class SingleColumnEvaluateDescriptionChain(LLMChain):
    prompt: PromptTemplate = EVAL_COL_DESC_PROMPT
    output_parser: PydanticOutputParser = EVAL_COL_DESC_PROMPT.output_parser


def extract_columns(db, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(get_table(db, inputs["table_name"]).columns.keys())


def extract_params(db, inputs: Dict[str, Any]):
    return {
        "extracted": [
            {
                "column_name": col,
                "column_extract": "\n".join(
                    [
                        "/*" + str(s) + "*/"
                        for s in pd.read_sql(
                            f"SELECT {col} from {inputs['table_name']} limit 10;",
                            db._engine,
                        )[col].tolist()
                    ]
                ),
                **inputs,
            }
            for col in extract_columns(db, inputs)
        ]
    }


class ColumnInfoInputsListExtractor(ExtractChain):
    db: SQLDatabase
    input_variables: List[str] = ["table_name"]

    @root_validator(pre=True)
    def initialize_transform(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "transform" not in values:
            values["transform"] = lambda x: extract_params(values["db"], x)
        return values


class EnhanceInputs(BaseModel):
    db: SQLDatabase

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["table_name"]

    def _get_additional_inputs(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        additional = {}
        if "table_info" not in inputs.keys():
            additional["table_info"] = self.db.get_table_info_no_throw(
                [inputs["table_name"]]
            )

        return additional

    def __call__(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        additional = self._get_additional_inputs(inputs)
        for k, v in additional.items():
            inputs[k] = v
        return inputs


class MultipleColumnDescriptionChain(ParallelChain):
    llm: BaseLanguageModel
    db: SQLDatabase
    extract_inputs: ColumnInfoInputsListExtractor
    chain: Chain
    input_prepper: Optional[EnhanceInputs]
    output_key: str = "column_evaluations"

    @root_validator(pre=True)
    def initialize_input_prepper(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "input_prepper" not in values:
            values["input_prepper"] = EnhanceInputs(db=values["db"])
        return values

    @root_validator(pre=True)
    def initialize_extract_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "extract_inputs" not in values:
            values["extract_inputs"] = ColumnInfoInputsListExtractor(db=values["db"])
        return values

    @root_validator(pre=True)
    def initialize_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "chain" not in values:
            values["chain"] = SingleColumnEvaluateDescriptionChain(llm=values["llm"])
        return values

    @property
    def input_keys(self) -> List[str]:
        return ["table_name"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

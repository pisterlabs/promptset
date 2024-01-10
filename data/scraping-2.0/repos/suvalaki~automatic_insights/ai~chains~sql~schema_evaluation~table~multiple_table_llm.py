from typing import List, Dict, Optional, Any, Tuple

from pydantic import root_validator, BaseModel

from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain
from langchain.sql_database import SQLDatabase
from langchain.output_parsers import PydanticOutputParser

from ai.chains.sql.schema_evaluation.table.table_llm import TableEvaluationChain
from ai.chains.parallel_chain import ExtractChain, ParallelChain


class SingleExtractInputs(BaseModel):
    input_key: str = "tables"


class ExpandTableNameExtractor(ExtractChain, SingleExtractInputs):
    @root_validator(pre=True)
    def initialize_input_variables(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "input_key" in values:
            values["input_variables"] = [values["input_key"]]
        else:
            values["input_variables"] = []
        return values

    @root_validator
    def initialize_input_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["input_variables"] = [values["input_key"]]
        return values

    @root_validator(pre=True)
    def initialize_transform(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "transform" not in values:
            values["transform"] = lambda inputs: {
                "extracted": [{"table_name": v} for v in inputs[values["input_key"]]]
            }
        return values


class MultipleTableEvaluatorChain(ParallelChain):
    chain: TableEvaluationChain | Chain
    extract_inputs: ExpandTableNameExtractor = ExpandTableNameExtractor()
    output_key: str = "table_schema_evaluations"

    @property
    def input_keys(self) -> List[str]:
        return [self.extract_inputs.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

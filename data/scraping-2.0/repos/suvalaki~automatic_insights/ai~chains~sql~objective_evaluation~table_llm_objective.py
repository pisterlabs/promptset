from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser

from ai.chains.sql.schema_evaluation.table.table_llm import TableEvaluatedDescription
from ai.chains.sql.objective_evaluation.name_evaluation.base import (
    TableSelectionDetailThought,
)


class TableObjectiveEvaluation(TableSelectionDetailThought):
    ...


OBJ_EVAL_TABLE_DESC_TEMPLATE = (
    "You will be presented with an objective, a table schema, and "
    "some guesses about what each column in the table contains. "
    "You are to evaluate what the table in relation to the objective."
    "Be factual and specific. The claims about SQL should not make inferences "
    "that arent supported strongly by the inputs. Talk about summaries and "
    "filters and not counting entries in the table."
    "\n\nObjective: {objective}"
    "\n\nTablename: {table_name}"
    "\n\nInfered Column Descriptions: {infered_columns}"
    "\n\n{format_instructions}"
)


OBJ_EVAL_TABLE_DESC_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=TableObjectiveEvaluation
)

OBJ_EVAL_TABLE_DESC_PROMPT = PromptTemplate(
    template=OBJ_EVAL_TABLE_DESC_TEMPLATE,
    input_variables=["objective", "table_name", "infered_columns"],
    partial_variables={
        "format_instructions": OBJ_EVAL_TABLE_DESC_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=OBJ_EVAL_TABLE_DESC_OUTPUT_PARSER,
)


def create_objective_evaluate_table_chain(**kwargs):
    return LLMChain(
        prompt=OBJ_EVAL_TABLE_DESC_PROMPT,
        **kwargs,
        output_parser=OBJ_EVAL_TABLE_DESC_OUTPUT_PARSER,
        output_key="table_objective_eval"
    )

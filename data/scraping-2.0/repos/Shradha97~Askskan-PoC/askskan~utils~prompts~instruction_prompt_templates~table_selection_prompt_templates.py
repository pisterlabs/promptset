from langchain.prompts.prompt import PromptTemplate
from utils.prompts.table_selection_prompts.zero_shot_prompt import (
    zero_shot_prompt,
)
from langchain.output_parsers import CommaSeparatedListOutputParser

common_input_variables = [
    "table_metadata",
    "question",
    "examples",
]

MULTIPLE_TABLES_SELECTION_PROMPT = PromptTemplate(
    template=zero_shot_prompt,
    input_variables=common_input_variables,
    output_parser=CommaSeparatedListOutputParser(),
)

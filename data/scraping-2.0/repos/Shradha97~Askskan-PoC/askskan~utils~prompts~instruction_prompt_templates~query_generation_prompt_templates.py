from langchain.prompts.prompt import PromptTemplate
from utils.prompts.query_generation_prompts.few_shot_prompt import (
    few_shot_prompt_with_additional_instruction_with_persona,
    few_shot_prompt_without_additional_instruction_with_persona,
    few_shot_prompt_with_additional_instruction_without_persona,
    few_shot_prompt_without_additional_instruction_without_persona,
)

common_input_variables = [
    "context",
    "data_table_name",
    "question",
    "chat_history",
    "schema_definitions",
    "start_date",
    "end_date",
    "examples",
]

QUERY_GENERATION_PROMPT_WITH_ADDITIONAL_INS_WITH_PERSONA = PromptTemplate(
    template=few_shot_prompt_with_additional_instruction_with_persona,
    input_variables=common_input_variables + ["persona_id", "additional_instructions"],
)

QUERY_GENERATION_PROMPT_WITH_ADDITIONAL_INS_WITHOUT_PERSONA = PromptTemplate(
    template=few_shot_prompt_with_additional_instruction_without_persona,
    input_variables=common_input_variables + ["additional_instructions"],
)
QUERY_GENERATION_PROMPT_WITHOUT_ADDITIONAL_INS_WITH_PERSONA = PromptTemplate(
    template=few_shot_prompt_without_additional_instruction_with_persona,
    input_variables=common_input_variables + ["persona_id"],
)

QUERY_GENERATION_PROMPT_WITHOUT_ADDITIONAL_INS_WITHOUT_PERSONA = PromptTemplate(
    template=few_shot_prompt_without_additional_instruction_without_persona,
    input_variables=common_input_variables,
)

from langchain.prompts import PromptTemplate

TRANSLATOR_PROMPT_TMPL = (
    "CONSTRAINTS:\n"
    "Your task is to translate the given text into Korean, keeping in mind the following instructions:\n"
    "1. Translate the provided text into Korean, ensuring a natural and fluent representation of the message.\n"
    "2. Exclude function names, variable names, and any coding-related terminology from the translation.\n"
    "3. Maintain any terms or expressions enclosed in double quotes in their original form without translation.\n"
    "4. Your target audience is programmers, so the translated text should be easily understandable to them.\n"
    "If no text is provided, respond with '없습니다.'\n"
    "TEXT TO TRANSLATE:\n"
    "--------\n"
    "{text}\n"
    "--------\n"
)

TRAMSLATOR_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=TRANSLATOR_PROMPT_TMPL,
)
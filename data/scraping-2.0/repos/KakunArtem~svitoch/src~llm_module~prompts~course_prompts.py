from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from src.llm_module.chains.models.course_model import Course

course_model = PydanticOutputParser(pydantic_object=Course)

advance_course_template = """
### Instructions ###
If MUST ONLY generate a step-by-step self-study course based on the above topics".
You MUST ALWAYS answer in literature {language} language even if the user questions in another language.
You MUST give very precise instructions. 
You MUST include Topic-related terms in instructions.
You MUST be specific, do not give general advice.
You MUST follow the RULES within the section if provided.
### Instructions ###

### Format ###
{format_instructions}
### Format ###

Topics: '''{query}'''
"""

course_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(advance_course_template)
    ],
    input_variables=["query", "language"],
    partial_variables={
        "format_instructions": course_model.get_format_instructions()
    }
)

lessons_template = """
### Instructions ###
If MUST ONLY generate a comprehensive guide for each of the provided topics.
You MUST provide detailed code examples with explanations.
You MUST provide comparison tables.
You MUST use a detailed explanation.
You MUST return a response as a formatted Markdown text.
YOU MUST consider the provided Context in your answer.
You MUST ALWAYS answer in literature {language} language even if the user questions in another language.
### Instructions ###

### Context ####
{context}
### Context ####

Lessons: '''{lessons}'''
"""

lessons_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(lessons_template)
    ],
    input_variables=["lessons", "language", "context"]
)
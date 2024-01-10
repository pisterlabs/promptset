import re

from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List 2 interview question related to {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)

def filter_newlines_and_convert_to_list(output):
    """Replace newline characters with comma-space and convert the resulting string into a list."""
    if isinstance(output, str):
        filtered_output = output.replace('\n', ', ')
        return filtered_output.split(', ')
    elif isinstance(output, list):
        filtered_list = []
        for item in output:
            filtered_output = item.replace('\n', ', ')
            filtered_list.extend(filtered_output.split(', '))
        return filtered_list
    return output  # If not a string or list, return the original output unchanged.

def strip_leading_numbers_from_list(items):
    """Strip leading numbers and punctuation from each item in the list."""
    stripped_items = []
    for item in items:
        # Using regex to remove any leading numbers followed by punctuation (like '.')
        stripped = re.sub(r"^\d+\.\s*", "", item)
        stripped_items.append(stripped)
    return stripped_items


def generate_interview_questions_for_subject(subject):
    _input = prompt.format(subject=subject)
    output = model(_input)
    parsed_output = output_parser.parse(output)
    return filter_newlines_and_convert_to_list(parsed_output)

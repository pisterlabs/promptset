from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader

from chat_with_x.utils.dataclasses import TaskDescriptions, InstructionDescriptions

import textwrap
import re
import os


# web/beautiful_soup_web
# web/knowledge_base
# web/readability_web
# web/rss
# web/simple_web
# web/unstructured_web


class DataLoaderMapper:
    def __init__(self):
        self.data_class_map = {
            'TXT': UnstructuredFileLoader,
            'RTF': UnstructuredFileLoader,
            'ODT': UnstructuredFileLoader,
            'DOC': UnstructuredWordDocumentLoader,
            'DOCX': UnstructuredWordDocumentLoader,
            'PDF': PyPDFLoader,
            'MD': UnstructuredMarkdownLoader,
            'JSON': UnstructuredFileLoader,
            'CSV': CSVLoader,
            'PPT': UnstructuredPowerPointLoader,
            'PPTX': UnstructuredPowerPointLoader,
            'HTML': UnstructuredHTMLLoader,
            'DIR': DirectoryLoader,
        }

    def ext_to_lc_data_loader(self, file_extension):
        """
        Returns the appropriate data loader class for the given file extension
        and falls back to the UnstructuredFileLoader in cases where the file extension
        is not recognized.

        Args:
            file_extension (str): The file extension to be used to determine the appropriate data loader class

        Returns:
            The appropriate data loader class
        """
        return self.data_class_map.get(file_extension, UnstructuredFileLoader)

    def get_loader_from_path(self, f_path):
        """ Returns the appropriate data loader class for the given file path

        Args:
            f_path (str): The path to the file to be loaded

        Returns:
            The appropriate data loader class from the Langchain library
        """
        file_ext_pattern = r'\.([a-zA-Z0-9]+)$'
        file_ext_match = re.search(file_ext_pattern, f_path)
        if file_ext_match:
            file_ext = file_ext_match.group(1).upper()
        elif os.path.isdir(f_path):
            file_ext = 'DIR'
        else:
            print(f"Could not determine file extension for file: {f_path}. Falling back to UnstructuredFileLoader")
            file_ext = 'UNKNOWN'
        return self.ext_to_lc_data_loader(file_ext)(f_path)


def file_helper_prompt(one_line_desc,
                       task_type="initial",
                       instruction="conversational",
                       prevent_uncertainty=True,
                       restrict_answer_to_context=False):
    """ Generates a prompt for the file helper task

    Args:
        one_line_desc (str): A one line description of the document to be used in the prompt.
        task_type (str): The type of task to be performed. This is used to generate the task description.
        instruction (str): The type of instruction to be used in the prompt. This is used to generate the instruction text.
        prevent_uncertainty (bool): Whether or not to include the uncertainty restriction in the prompt.
        restrict_answer_to_context (bool): Whether or not to include the context restriction in the prompt.


    """

    # Define common task and instruction descriptions for perusal
    task_descriptions = TaskDescriptions().__dict__
    instruction_descriptions = InstructionDescriptions().__dict__

    # Get the task description and instruction from dictionaries or use the provided strings (if key not found)
    task_description = task_descriptions[task_type] if task_type in task_descriptions else task_type
    instruction_text = instruction_descriptions[instruction] if instruction in instruction_descriptions else instruction
    if "{one_line_desc}" in task_description:
        task_description = task_description.format(one_line_desc=one_line_desc)
    if "{one_line_desc}" in instruction_text:
        instruction_text = instruction_text.format(one_line_desc=one_line_desc)

    template = textwrap.dedent("""\
    You are an AI assistant for {task_description}
    You are given the following extracted parts of a document and input from a user. {instruction_text}
    {_uncertainty_line}{_context_restrict}
    User Input: {question}
    =========
    {context}
    =========
    Answer in Markdown:""")

    # Format the template with optional inputs
    if prevent_uncertainty:
        _uncertainty_line = "If you don't know the answer or how to answer, just say '???' Do not make up an answer.\n"
    else:
        _uncertainty_line = ""

    if restrict_answer_to_context:
        _context_restrict = f"If the input is not related to the given document, explain that you " \
                            f"are instructed to only answer queries within the context of the specific document.\n"
    else:
        _context_restrict = ""

        # Format the template with all inputs
    template = template.format(
        task_description=task_description,
        one_line_desc=one_line_desc,
        instruction_text=instruction_text,
        _uncertainty_line=_uncertainty_line,
        _context_restrict=_context_restrict,
        question="{question}",
        context="{context}"
    )
    return template

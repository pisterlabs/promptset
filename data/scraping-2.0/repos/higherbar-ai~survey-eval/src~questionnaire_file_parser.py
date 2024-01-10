#  Copyright (c) 2023 Higher Bar AI, PBC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utility functions for parsing questionnaire files."""

from questionnaire_file_reader import (read_docx, read_html, read_pdf_combined, parse_xlsx,
                                       read_local_html, parse_csv, split_langchain)
from typing import Optional, Callable
import os
import tempfile
import logging
import json
import asyncio
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from kor.extraction import create_extraction_chain
from kor import from_pydantic
from kor.nodes import Object
from kor.validators import Validator
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

# initialize global resources
parser_logger = logging.getLogger(__name__)


def create_schema(question_id_spec: str = None, module_spec: str = None, module_desc_spec: str = None,
                  question_spec: str = None,
                  instructions_spec: str = None, options_spec: str = None, language_spec: str = None,
                  kor_general_spec: str = None) -> tuple[Object, Validator]:
    """
    Create a schema based on a pydantic model for questionnaire data.

    This function generates a schema using dynamic descriptions for the fields of a Question class,
    and returns a schema and an extraction validator.

    :param question_id_spec: Specification for the 'question_id' field.
    :type question_id_spec: str
    :param module_spec: Specification for the 'module' field.
    :type module_spec: str
    :param module_desc_spec: Specification for the 'module_description' field.
    :type module_desc_spec: str
    :param question_spec: Specification for the 'question' field.
    :type question_spec: str
    :param instructions_spec: Specification for the 'instructions' field.
    :type instructions_spec: str
    :param options_spec: Specification for the 'options' field.
    :type options_spec: str
    :param language_spec: Specification for the 'language' field.
    :type language_spec: str
    :param kor_general_spec: Overall specification for the schema.
    :type kor_general_spec: str
    :return: A tuple containing the schema and extraction validator.
    :rtype: tuple[Object, Validator]
    """

    # set defaults as needed
    if not question_id_spec:
        question_id_spec = ('Question ID: a numeric or alphanumeric identifier or short variable name identifying a '
                            'specific question, usually located just before or at the beginning of a new question.')
    if not module_spec:
        module_spec = ('Module title: Represents the main section or category within which a series of questions are '
                       'located (e.g., "Health" or "Demographics"). It might include a number or index, but should '
                       'also include a short title.')
    if not module_desc_spec:
        module_desc_spec = ("Module introduction: Introductory text or instructions that appear at the start of a new "
                            "module, before the module's questions appear.")
    if not question_spec:
        question_spec = ('Question: A single question or label/description of a single form field, often following '
                         'a numerical code or identifier like "2.01." or "gender:" Must be text designed to elicit '
                         'specific information, often in the form of a question (e.g., "How old are you?") or prompt '
                         '(e.g., "Your age:"). Might be in different languages, but the structure remains the same.')
    if not instructions_spec:
        instructions_spec = ('Question instructions: Instructions or other guidance about how to ask or answer the '
                             'question, including enumerator or interviewer instructions. If the question includes '
                             'a list of specific response options, do NOT include those in the instructions.')
    if not options_spec:
        options_spec = ("Question options: The list of specific response options for multiple-choice questions. "
                        "Often listed immediately after the question or instructions. Might include numbers, "
                        "letters, or specific codes followed by the actual response option text. Separate options "
                        "with a space, a pipe symbol, and another space, like this: '1. Yes | 2. No'.")
    if not language_spec:
        language_spec = 'Question language: The language in which the question is written.'
    if not kor_general_spec:
        kor_general_spec = ('Questionnaire: A questionnaire consists of a list of questions or prompts (question) '
                            'that are used to collect data from respondents. Each question might include a short ID '
                            'number or name (question_id), instructions, and/or a list of specific response options '
                            '(options), and each question might appear in multiple languages (language). These '
                            'questions might be organized within a series of modules (or sections), each of which '
                            'might have a title and introductory instructions '
                            '(module_description). You must return the questionnaire in the '
                            'same order as it was given to you and in each json you must return either a module '
                            'or question. If there is a question that is not complete, DO NOT return it.')

    class QuestionnaireData(BaseModel):
        """
        Pydantic model representing a questionnaire question.

        Each field of the model is optional and includes a description provided
        as a parameter to the create_schema function.
        """

        question_id: Optional[str] = Field(description=question_id_spec)
        module: Optional[str] = Field(description=module_spec)
        module_description: Optional[str] = Field(description=module_desc_spec)
        question: Optional[str] = Field(description=question_spec)
        instructions: Optional[str] = Field(description=instructions_spec)
        options: Optional[str] = Field(description=options_spec)
        language: Optional[str] = Field(description=language_spec)

    # generate schema and extraction validator from the QuestionnaireData class
    schema, extraction_validator = from_pydantic(
        QuestionnaireData,
        description=kor_general_spec,
        examples=[
            ("""2. Demographics

We’ll begin with some questions so that we can get to know you and your family.

[BIRTHYR] What year were you born?

[GENDER] Which gender do you identify with?

Female

Male

Non-binary

Prefer not to answer

[ZIPCODE] What is your zip code?""", {"questionnairedata": [
                {
                    "module": "2. Demographics",
                    "module_description": "We’ll begin with some questions so that we can get to know you and your "
                                          "family.",
                },
                {
                    "question_id": "BIRTHYR",
                    "question": "What year were you born?",
                    "language": "English"
                },
                {
                    "question_id": "GENDER",
                    "question": "Which gender do you identify with?",
                    "options": "Female | Male | Non-binary | Prefer not to answer",
                    "language": "English"
                },
                {
                    "question_id": "ZIPCODE",
                    "question": "What is your zip code?",
                    "language": "English"
                }
            ]}),
            ("""[EDUCATION] What is the highest level of education you have completed?

o Less than high school

o High school / GED

o Some college

o 2-year college degree

o 4-year college degree

o Vocational training

o Graduate degree

o Prefer not to answer""", {"questionnairedata": [
                {
                    "question_id": "EDUCATION",
                    "question": "What is the highest level of education you have completed?",
                    "options": "Less than high school | High school / GED | Some college | 2-year college degree | "
                               "4-year college degree | Vocational training | Graduate degree | Prefer not to answer",
                    "language": "English"
                }
            ]}),
            ("""And how much do you disagree or agree with the following statements? For each statement, please """
             """rate how much the pair of traits applies to you, even if one trait applies more strongly than the """
             """other. I see myself as... [For each: 1=Strongly disagree, 7=Strongly agree]

[BIG5Q1] Extraverted, enthusiastic

[BIG5Q2] Critical, quarrelsome

[BIG5Q3] Dependable, self-disciplined""", {"questionnairedata": [
                {
                    "question_id": "BIG5Q1",
                    "question": "And how much do you disagree or agree with the following statement? I see myself "
                                "as... Extraverted, enthusiastic",
                    "options": "1=Strongly disagree | 2 | 3 | 4 | 5 | 6 | 7=Strongly agree",
                    "instructions": "Please rate how much the pair of traits applies to you, even if one trait applies "
                                    "more strongly than the other.",
                    "language": "English"
                },
                {
                    "question_id": "BIG5Q2",
                    "question": "And how much do you disagree or agree with the following statement? I see myself "
                                "as... Critical, quarrelsome",
                    "options": "1=Strongly disagree | 2 | 3 | 4 | 5 | 6 | 7=Strongly agree",
                    "instructions": "Please rate how much the pair of traits applies to you, even if one trait applies "
                                    "more strongly than the other.",
                    "language": "English"
                }]}),
            (
                """4. Savings Habits

Next, we will ask questions over your monthly saving habits and the potential methods that are used to save money.

1. On average, how much money do you spend monthly on essential goods below that contribute to your wellbeing """
                """(explain/ add in an example)

2. How do you typically spend your monthly income? (choose all that may apply)

a. Home and Housing

b. Retirement

c. Bills and Utility

d. Medical (Physical and Mental Treatment and Care)

e. Taxes

f. Insurance

g. Credit Card Payments (if applicable)

h. Food

i. Shopping and personal items

j. Other

k. I am not able to save money each month

l. Nothing

m. Don’t Know

3. Do you contribute the same amount or more to your savings each month?""", {"questionnairedata": [
                    {
                        "module": "4. Savings Habits",
                        "module_description": "Next, we will ask questions over your monthly saving habits and the "
                                              "potential methods that are used to save money.",
                    },
                    {
                        "question_id": "1",
                        "question": "On average, how much money do you spend monthly on essential goods below that "
                                    "contribute to your wellbeing (explain/ add in an example)",
                        "language": "English"
                    },
                    {
                        "question_id": "2",
                        "question": "How do you typically spend your monthly income?",
                        "options": "a. Home and Housing | b. Retirement | c. Bills and Utility | d. Medical "
                                   "(Physical and Mental Treatment and Care) | e. Taxes | f. Insurance | g. Credit "
                                   "Card Payments (if applicable) | h. Food | i. Shopping and personal items | j. "
                                      "Other | k. I am not able to save money each month | l. Nothing | m. Don’t Know",
                        "instructions": "(choose all that may apply)",
                        "language": "English"
                    },
                    {
                        "question_id": "3",
                        "question": "Do you contribute the same amount or more to your savings each month?",
                        "language": "English"
                    }
                ]}
            ),
            (
                """<h1>Round 1, June 2020 Eng</h1>
<table border="1" class="dataframe">
<tbody>
<tr>
<td>Module</td>
<td>Section</td>
<td>Variable</td>
<td>Question</td>
<td>Response set</td>
</tr>
<tr>
<td></td>
<td>CONS. Introduction and Consent</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>CONS</td>
<td></td>
<td>Good morning/afternoon/evening. My name is ______________________ from Innovations from Poverty Action, a """
                """Mexican research NGO. \n \n We would like to invite you to participate in a survey lasting about """
                """20 minutes about the effects of covid-19 on economic and social conditions in the Mexico City """
                """metropolitan area. If you are eligible for the survey we will compensate you [30 pesos] in """
                """airtime for completing your survey.</td>
<td></td>
</tr>
<tr>
<td></td>
<td>CONS</td>
<td>cons1</td>
<td>Can I give you more information?</td>
<td>Y/N</td>
</tr>
<tr>
<td></td>
<td>CONS</td>
<td></td>
<td>*If cons1=N\n Thank you for your response. We will end the survey now.</td>
<td>[End survey]</td>
</tr>
<tr>
<td>core</td>
<td>END</td>
<td>end4</td>
<td>What is your first name?</td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>DEM</td>
<td>dem1</td>
<td>How old are you?</td>
<td>*Enter age*\n ###</td>
</tr>
<tr>
<td>core</td>
<td>CONS</td>
<td></td>
<td>*If DEM1&lt;18*\n Thank you for your response. We will end the survey now.</td>
<td>[End survey]</td>
</tr>""", {"questionnairedata": [
                    {
                        "module": "CONS. Introduction and Consent",
                    },
                    {
                        "question": """Good morning/afternoon/evening. My name is ______________________ from """
                        """Innovations from Poverty Action, a """
                        """Mexican research NGO. \n \n We would like to invite you to participate in a survey """
                        """lasting about 20 minutes about the effects of covid-19 on economic and social """
                        """conditions in the Mexico City metropolitan area. If you are eligible for the survey """
                        """we will compensate you [30 pesos] in airtime for completing your survey.""",
                        "language": "English"
                    },
                    {
                        "question_id": "cons1",
                        "question": "Can I give you more information?",
                        "options": "Y | N",
                        "instructions": "*If cons1=N\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    },
                    {
                        "question_id": "end4",
                        "question": "What is your first name?",
                        "language": "English"
                    },
                    {
                        "question_id": "dem1",
                        "question": "How old are you?",
                        "instructions": "*Enter age*\n ###\n"
                                        "*If DEM1&lt;18*\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    }
                ]}
            ),
            (
                """<tr>
<td>MEXICO</td>
<td>INC</td>
<td>inc11_mex</td>
<td>*If YES to INC12_mex*\n If schools and daycares remained closed and workplaces re-opened, would anyone in your """
                """household have to stay home and not return to work in order to care for children too young to """
                """stay at home without supervision?</td>
<td>*Read out, select multiple possible*\n Grandparents\n Hired babysitter\n Neighbors\n Mother who normally st\n """
                """Mother who normally works outside the home\n Father who normally works outside the home\n Older """
                """sibling\n DNK</td>
</tr>
<tr>
<td></td>
<td>NET. Social Safety Net</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>NET</td>
<td>net1</td>
<td>Do you usually receive a regular transfer from any cash transfer or other in-kind social support program?\n """
                """\n HINT: Social safety net programs include cash transfers and in-kind food transfers (food """
                """stamps and vouchers, food rations, and emergency food distribution). Example includes XXX cash """
                """transfer programme.</td>
<td>Y/N/DNK</td>
</tr>""", {"questionnairedata": [
                    {
                        "question_id": "inc11_mex",
                        "question": """If schools and daycares remained closed and workplaces re-opened, would """
                        """anyone in your household have to stay home and not return to work in order to care for """
                        """children too young to stay at home without supervision?""",
                        "options": "Grandparents | Hired babysitter | Neighbors | Mother who normally st | Mother who "
                                   "normally works outside the home | Father who normally works outside the home | "
                                   "Older sibling | DNK",
                        "instructions": "*If YES to INC12_mex*\n*Read out, select multiple possible*",
                        "language": "English"
                    },
                    {
                        "module": "NET. Social Safety Net",
                    },
                    {
                        "question_id": "net1",
                        "question": """Do you usually receive a regular transfer from any cash transfer or other """
                                    """in-kind social support program?\n \n HINT: Social safety net programs """
                                    """include cash transfers and in-kind food transfers (food stamps and vouchers, """
                                    """food rations, and emergency food distribution). Example includes XXX cash """
                                    """transfer programme.""",
                        "options": "Y | N | DNK",
                        "instructions": "*If cons1=N\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    }
                ]}
            ),
            (
                """<tr>
<td></td>
<td>POL. POLICING</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>MEXICO</td>
<td>POL</td>
<td></td>
<td>Now I am going to ask you some questions about the main problems of insecurity in Mexico City and the """
                """performance of the city police since the coronavirus pandemic began around March 20, 2020.</td>
<td></td>
</tr>
<tr>
<td>MEXICO</td>
<td>POL</td>
<td>POL1</td>
<td>Compared to the level of insecurity that existed in your neighborhood before the pandemic began, do you """
                """consider that the level of insecurity in your neighborhood decreased, remained more or less the """
                """same, or increased?</td>
<td>Decreased\n it was more or less the same\n increased\n (777) Doesn’t answer\n (888) Doesn’t know\n (999) """
                """Doesn’t apply</td>
</tr>""", {"questionnairedata": [
                    {
                        "module": "POL. POLICING",
                        "module_description": "Now I am going to ask you some questions about the main problems of "
                                              "insecurity in Mexico City and the performance of the city police since "
                                              "the coronavirus pandemic began around March 20, 2020."
                    },
                    {
                        "question_id": "POL1",
                        "question": """Compared to the level of insecurity that existed in your neighborhood before """
                                    """the pandemic began, do you consider that the level of insecurity in your """
                                    """neighborhood decreased, remained more or less the same, or increased?""",
                        "options": "Decreased | it was more or less the same | increased | (777) Doesn’t answer | "
                                   "(888) Doesn’t know | (999) Doesn’t apply",
                        "language": "English"
                    }
                ]}
            )
        ],
        many=True,
    )

    return schema, extraction_validator


def generate_extractor_chain(model_input: str, api_base: str, openai_api_key: str, open_api_version: str,
                             schema: Object, default_kor_prompt: str = None, provider: str = "azure") -> LLMChain:
    """
    Generate an extractor chain based on the specified language model and API settings.

    :param model_input: Name of the language model to use.
    :type model_input: str
    :param api_base: Base URL for the API.
    :type api_base: str
    :param openai_api_key: API key for accessing OpenAI services.
    :type openai_api_key: str
    :param open_api_version: Version of the OpenAI API to use.
    :type open_api_version: str
    :param schema: Schema definition for the extraction.
    :type schema: Object
    :param default_kor_prompt: Default prompt template for knowledge ordering and reasoning.
    :type default_kor_prompt: str
    :param provider: Provider for the LLM service ("openai" for direct OpenAI, "azure" for Azure). Default is "azure".
    :type provider: str
    :return: An extraction chain configured with the specified parameters.
    :rtype: LLMChain
    """

    # set defaults as needed
    if not default_kor_prompt:
        default_kor_prompt = ("Your goal is to extract structured information from the user's input that matches "
                              "the format described below. When extracting information, please make sure it matches "
                              "the type information exactly. Please return the information in order. Only extract the "
                              "information that is in the document. Do not add any extra information. Do not add any "
                              "attributes that do not appear in the schema shown below.\n\n"
                              "{type_description}\n\n{format_instructions}\n\n")

    # initialize LLM
    if provider == "azure":
        llm = AzureChatOpenAI(
            temperature=0,
            verbose=True,
            model_name=model_input,
            openai_api_base=api_base,
            openai_api_version=open_api_version,
            deployment_name=model_input,
            openai_api_key=openai_api_key,
            openai_api_type="azure",
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            temperature=0,
            verbose=True,
            model_name=model_input,
            openai_api_key=openai_api_key
        )
    else:
        raise ValueError("Unsupported provider specified. Choose 'openai' or 'azure'.")

    # define the prompt template for the extraction chain
    template = PromptTemplate(
        input_variables=["type_description", "format_instructions"],
        template=default_kor_prompt,
    )

    # create and return the extraction chain
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="JSON",
                                    instruction_template=template, input_formatter="triple_quotes")
    return chain


def uri_validator(url: str) -> bool:
    """
    Validate if the given string is a valid URI.

    :param url: The string to validate as URI.
    :type url: str
    :return: True if the string is a valid URI, False otherwise.
    :rtype: bool
    """

    result = urlparse(url)
    return all([result.scheme, result.netloc])


def get_data_from_url(url: str, splitter: Callable = split_langchain) -> list[str] | dict:
    """
    Fetch and process data from a given URL based on file extension.

    :param url: URL of the file to process.
    :type url: str
    :param splitter: Function or method used for processing the data.
    :type splitter: Callable
    :return: Processed data based on file extension or None if an error occurs.
    :rtype: list[str] | dict | None
    """

    valid_url = uri_validator(url)
    response = []
    if valid_url:
        response = requests.get(url.strip())
        if response.status_code != 200:
            parser_logger.log(logging.ERROR, f'Error: {response.status_code}')
            return []

    extension = url.split('.')[-1].strip().lower()
    process_functions = {
        'pdf': read_pdf_combined,
        'docx': read_docx,
        'csv': parse_csv,
        'xlsx': parse_xlsx,
        'html': read_local_html
    }

    if extension in process_functions:
        # process based on the extension
        process_function = process_functions.get(extension, read_html)
        if valid_url:
            # if the URL is valid, save the file locally
            temp_file_handle, temp_file_path = tempfile.mkstemp(suffix='.' + extension)
            with os.fdopen(temp_file_handle, 'wb') as temp_file:
                temp_file.write(response.content)
            file_to_process = temp_file_path
        else:
            file_to_process = url
        retval = process_function(file_to_process, splitter)
    else:
        # for other extensions, assume it's a URL to an HTML page
        retval = read_html(url, splitter)

    # if the result is a list of documents, extract the page content for a list of strings
    if isinstance(retval, list) and retval:
        retval = [doc.page_content if not isinstance(doc, str) else doc for doc in retval]
    return retval


def process_kor_data(data: dict) -> list:
    """
    Process and structure data specifically for knowledge ordering and reasoning.

    :param data: Data to be processed, expected to have a 'questionnairedata' field.
    :type data: dict
    :return: A list of structured data.
    :rtype: list
    """

    # log raw data for debugging purposes
    parser_logger.log(logging.DEBUG, f"Raw data:\n{json.dumps(data, indent=2)}")

    # handle either 1 or 2 layers of outer nesting
    inner_data = data.get('questionnairedata', [])
    if len(inner_data) == 1 and isinstance(inner_data[0], dict) and 'questionnairedata' in inner_data[0]:
        # if the data is nested, process the inner data
        inner_data = inner_data[0]['questionnairedata']

    return [record for record in inner_data]


async def safe_apredict(chain: LLMChain, page: str):
    """
    Asynchronously predict with error handling, ensuring a safe call to the AI prediction chain.

    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :param page: The input text to be processed.
    :type page: str
    :return: The prediction result or a default value in case of an error.
    """

    try:
        return await chain.apredict(text=page)
    except Exception as e:
        # report, then return a default value in case of an error
        parser_logger.log(logging.ERROR, f"An error occurred: {e}")
        return {'data': []}


def total_string_length(d: dict) -> int:
    """
    Calculate the total string length of all values in a dictionary.

    :param d: The dictionary whose values' string lengths are to be summed.
    :type d: dict
    :return: The total string length of all values.
    :rtype: int
    """

    return sum(len(str(value)) for value in d.values())


def clean_data(data: dict) -> dict:
    """
    Clean the provided data by removing empty modules and questions — and by removing duplicate questions,
    keeping the ones with more content.

    :param data: The data to be cleaned.
    :type data: dict
    :return: The cleaned data.
    :rtype: dict
    """

    cleaned_data = {}
    for key, dt in data.items():
        # assemble module questions, dropping empty questions
        module = {question: question_data for question, question_data in dt.items() if question and question_data}

        # skip empty modules
        if not module:
            parser_logger.log(logging.INFO, f"Skipping empty module: {key}")
        else:
            parser_logger.log(logging.INFO, f"* Module: {key}")
            # for each question, look for questions with the same question text
            for question, question_data in module.items():
                if question_data:
                    to_remove = set()
                    for i in range(len(question_data)):
                        for j in range(i + 1, len(question_data)):
                            if question_data[i]['question'] == question_data[j]['question']:
                                # compare total string lengths and mark the shorter one for removal
                                length_i = total_string_length(question_data[i])
                                length_j = total_string_length(question_data[j])
                                if length_i > length_j:
                                    to_remove.add(j)
                                else:
                                    to_remove.add(i)
                                parser_logger.log(logging.INFO, f"  * Dropping duplicate question with shorter "
                                                                f"content: "
                                                                f"{question} - {question_data[i]['question']} "
                                                                f"({length_i} vs {length_j})")

                    # Remove duplicates after identifying them
                    for index in sorted(to_remove, reverse=True):
                        question_data.pop(index)

            cleaned_data[key] = module

    return cleaned_data


async def extract_data(chain: LLMChain, url: str) -> dict:
    """
    Asynchronously process the content from a given URL and parse it into structured data.

    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :param url: URL of the document to process.
    :type url: str
    :return: Structured and cleaned data.
    :rtype: dict
    """

    docs = get_data_from_url(url)
    structured = []
    grouped_content = {}

    if isinstance(docs, list):
        # process list of documents asynchronously
        if docs:
            # track our LLM usage with an OpenAI callback
            with get_openai_callback() as cb:
                # create a list of tasks, then execute them asynchronously
                tasks = [safe_apredict(chain, page) for page in docs]
                results = await asyncio.gather(*tasks)

                # report LLM usage
                parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
                parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
                parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
                parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
                parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")

                # parse list of results
                for res in results:
                    # if the resulting data is a dict, process it
                    if isinstance(res['data'], dict):
                        structured.extend(process_kor_data(res['data']))
    else:
        # process single document, tracking our LLM usage with an OpenAI callback
        structured = process_kor_data(docs)

    # organize questions by module and question ID
    question_module = {}
    current_module = '(none)'
    unknown_id_count = 0
    for record in structured:
        # get module name, defaulting to the current one
        module = record.get('module', current_module)

        # process question, if any
        if record.get('question', '').strip():
            # get question ID, if available
            question_id = record.get('question_id', '')
            if not question_id:
                # if no question ID is provided, generate a unique ID
                unknown_id_count += 1
                question_id = f"unknown_id_{unknown_id_count}"

            # always keep questions with the same ID together in the same module
            if question_id in question_module:
                module = question_module[question_id]
            else:
                question_module[question_id] = module

            # add question, grouped by module and question ID
            grouped_content.setdefault(module, {}).setdefault(question_id, [])
            grouped_content[module][question_id].append({
                'question': record['question'],
                'language': record.get('language', ''),
                'options': record.get('options', ''),
                'instructions': record.get('instructions', ''),
            })

        # remember current module for next question
        current_module = module

    # return cleaned-up version of the data
    return clean_data(grouped_content)


async def extract_data_from_directory(path_to_ingest: str, chain: LLMChain) -> list:
    """
    Extract structured data from all files in a specified directory.

    :param path_to_ingest: Path to the directory containing files to process.
    :type path_to_ingest: str
    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :return: A list of structured data from all processed files.
    :rtype: list
    """

    data_list = []
    for root, dirs, files in os.walk(path_to_ingest):
        for file in files:
            if file.endswith((".DS_Store", ".db")):
                # skip system files
                continue

            file_path = os.path.join(root, file)
            data_list.append(await extract_data(chain, file_path))

    return data_list


async def extract_data_from_file(file_path: str, chain: LLMChain) -> dict:
    """
    Extract structured data from a single file.

    :param file_path: Path to the file to process.
    :type file_path: str
    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :return: Structured data extracted from the processed file.
    :rtype: dict
    """

    return await extract_data(chain, file_path)

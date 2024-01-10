import main
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts details of elected directors from the sorted pages of minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains the page number and file name of a sorted file.

    Returns:
        A list of dictionaries where each dictionary represents an elected officer and includes their full name, election
        date, address, title, and URL of the source document page where the officer's details were extracted from.
    """

    elected_directors = [{}]
    election_of_director_content = ""
    election_of_director_provenance = []
    election_of_director_token_count = 0
    election_of_director_max_token_limit = 1024
    extracting_election_of_director = False
    minimum_number_of_directors = []
    maximum_number_of_directors = []

    file_count = len(sorted_files)
    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()
        parsed_this_page = False

        #  "minimum_directors": string, // Minimum number of directors required for the corporation
        if ("minimum" in lowercase_content or "less than" in lowercase_content) and "directors" in lowercase_content and "number" in lowercase_content:
            min_directors = extract_minimum_directors(content)
            if min_directors is not None:
                minimum_number_of_directors.append({"min_directors": min_directors, "provenance": main.get_url(file_name)})

        #  "maximum_directors": string, // Maximum number of directors allowed for the corporation
        if ("maximum" in lowercase_content or "more than" in lowercase_content) and "directors" in lowercase_content and "number" in lowercase_content:
            max_directors = extract_maximum_directors(content)
            if max_directors is not None:
                maximum_number_of_directors.append({"max_directors": max_directors, "provenance": main.get_url(file_name)})

        #  "directors": array, // One or more directors of a corporation, with child properties for their full name, election date, and address
        if "elected" in lowercase_content and "director" in lowercase_content and "register" in lowercase_content:
            extracting_election_of_director = True

        if extracting_election_of_director is True:
            election_of_director_tokens = election_of_director_token_count + main.num_tokens_from_string(content)
            election_of_director_provenance.append(main.get_url(file_name))

            if election_of_director_tokens < election_of_director_max_token_limit:
                election_of_director_token_count += election_of_director_tokens
                election_of_director_content += content
                parsed_this_page = True

            if election_of_director_tokens >= election_of_director_max_token_limit or page_number == file_count:
                if parsed_this_page:
                    output = extract_election_of_directors(election_of_director_content)
                else:
                    output = extract_election_of_directors(content)

                if output is not None:
                    try:
                        output = json.loads(output)
                        found_directors = []
                        for director in elected_directors:
                            if not bool(director):
                                elected_directors.remove(director)

                        for item in output:
                            if "director_name" in director and director['director_name'] == item['director_name']:
                                director['director_name'] = item['director_name'].title()
                                director['date_elected'] = item['date_elected']
                                director['date_retired'] = item['date_retired']
                                if "address" in item and isinstance(item['address'], str):
                                    director['address'] = re.sub(r'\s+', ' ', item['address']).strip()
                                try:
                                    director['provenance'] = item['provenance']
                                except KeyError as e:
                                    print(e)
                            else:
                                item['director_name'] = item['director_name'].title()
                                item['address'] = main.extract_address_for_person(person=item['director_name'], sorted_files=sorted_files)
                                item['provenance'] = election_of_director_provenance
                                found_directors.append(item)                    

                        for director in found_directors:
                            if not any(d['director_name'] == director['director_name'] for d in elected_directors):
                                elected_directors.append(director)

                    except json.decoder.JSONDecodeError:
                        pass

                extracting_election_of_director = False
                election_of_director_content = ""
                election_of_director_token_count = 0
                election_of_director_provenance = []
                election_of_director_tokens = 0

    output = {"directors": elected_directors, "minimum_directors": minimum_number_of_directors, "maximum_directors": maximum_number_of_directors}
    return output


# The following functions use a large language model to perform question & answer-style extraction from a minute book


def extract_minimum_directors(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""What is the minimum number of directors who can sit on
                    the board of directors? If this passage is about quorum rules return Not Found.
                    Format output as a number.
                    Passage:
                    {content}
                    Minimum:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "Not Found":
        return output


def extract_maximum_directors(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""What is the maximum number of directors who can sit on
                    the board of directors? If this passage is about quorum rules return Not Found.
                    Format output as a number.
                    Passage:
                    {content}
                    Maximum:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "Not Found":
        return output


def extract_election_of_directors(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""List the names of the directors of the corporation, the
                    date they were elected, and the date they retired (if not a current director).
                    The output should be a JSON object with one or more children having the following schema:
                    {{
                    "director_name": string  // Name of the elected director
                    "date_elected": string  // Formatted date (YYYY-MM-DD) of the elected date
                    "date_retired": string  // Formatted date (YYYY-MM-DD) of the retired date
                    "address": string // Address of the elected director
                    }}
                    If the passage does not mention names of directors, output [].
                    Passage:
                    {content}
                    Directors JSON:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2,
                                  max_output_tokens=1024),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "[]":
        return output

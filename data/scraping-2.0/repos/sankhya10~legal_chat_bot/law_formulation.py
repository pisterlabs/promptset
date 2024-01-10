from bs4 import BeautifulSoup
import re
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from termcolor import colored
import ast
import openai
import streamlit as st

def check_api_key(api_key_input):
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            api_key=api_key_input,
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        return False
    else:
        return True
    
def create_llm(api_key_input):
    if check_api_key(api_key_input):
        global model
        try:
            response = openai.Completion.create(engine="gpt4",prompt="This is a test.",max_tokens=5)
        except:
            model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key_input)
            return True
        else:
            model = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=api_key_input)
            return True
    else:
        return False

def further_refine(query):
    prompt = f""" Paraphrase the following sentence:
                    Sentence :{query}
                    Emit a single new sentence in a list!
                """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=""),
    ]

    responses = model(messages)

    return responses

def outline_guided(keywords, query):
    prompt = f"""You are an expert in detecting whether a query is relevant to dictionary of keywords :{keywords}
    Is the following query relevant or not relevant ?

    Query: {query}
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Answer either relevant or not relevant"),
    ]

    responses = model(messages)

    return responses


def create_agent_schema(keywords, query):

    context = f"""You are an expert in finding whether a query is either personal-law or business-law based on this dictionary: {keywords} \
                Is the following query personal-law or business-law ?
                
                Query:{query}
                """

    messages = [SystemMessage(content=context), HumanMessage(content="")]

    responses = model(messages)

    return responses


def refine_keywords(lawyer_keyword, query):

    context = f""" You have a list:{lawyer_keyword} that contains dictionary elements

                Your only job is extract the whole element (along with any values available) based on {query}. 
                That ELEMENT SHOULD BE INSIDE THE LIST !! YOU CAN"T OUTPUT ANY CODE RESULTS!!
                """

    messages = [SystemMessage(content=context), HumanMessage(content="")]

    responses = model(messages)

    return responses


def select_value(extracted, query):

    context = f""" You are an expert in determining a single best value out of {extracted} values  Relevant to query:{query}
        THE OUTPUT SHOULD BE A SINGLE VALUE FROM THIS {extracted} NOTHING ELSE THAN THAT!!
        """

    messages = [SystemMessage(content=context), HumanMessage(content="")]

    responses = model(messages)

    return responses


def create_lawyer_dict(file_path):
    pattern = r"personal-law_([\S]+)"

    with open(file_path, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Parse the HTML
    # soup = BeautifulSoup(html_data, 'html.parser')

    # Initialize dictionaries for business law and personal law
    law_dict = {"business-law": [], "personal-law": []}

    check_list = []
    # Find all <option> elements
    options = soup.find_all("option")

    # Process the options and organize into dictionaries
    law_dict = {"business-law": [], "personal-law": []}

    type_check = {}
    type_alo = {}
    for option in options:
        class_values = option.get("class", [])
        law_value = option["value"]
        # print(class_values,law_value)
        if "business-law" in class_values[0]:
            parts = class_values[0].split("_")
            if len(parts) >= 2 and parts[0] == "business-law":
                extracted_part = parts[1]
            if extracted_part not in law_value:
                type_check[f"{extracted_part}"] = []
        elif "personal-law" in class_values[0]:
            clever = option.text
            match = re.search(pattern, class_values[0])
            if match:
                extracted_part = match.group(1)
                if "-personal-law" in extracted_part:
                    extracted_part = extracted_part.replace("-personal-law", "")
            if "You have selected" in option.text:
                clever = clever.replace("You have selected", "")

            if law_value not in class_values[0]:
                type_alo[extracted_part] = []


    for option in options:
        class_values = option.get("class", [])
        law_value = option["value"]
        # print(class_values,law_value)
        if "business-law" in class_values[0]:
            parts = class_values[0].split("_")
            if len(parts) >= 2 and parts[0] == "business-law":
                extracted_part = parts[1]
                # print(extracted_part)
            if extracted_part in law_value:
                law_dict["business-law"].append({f"{law_value}"})
            else:
                if extracted_part in type_check.keys():
                    type_check[f"{extracted_part}"].append(law_value)

        elif "personal-law" in class_values[0]:
            clever = option.text
            match = re.search(pattern, class_values[0])
            if match:
                extracted_part = match.group(1)
                if "-personal-law" in extracted_part:
                    extracted_part = extracted_part.replace("-personal-law", "")
            if "You have selected" in option.text:
                clever = clever.replace("You have selected", "")

            if law_value in class_values[0]:
                law_dict["personal-law"].append({f"{clever}"})
            else:
                if extracted_part in type_alo.keys():
                    type_alo[extracted_part].append(clever)

    law_dict["business-law"].append(type_check)
    law_dict["personal-law"].append(type_alo)

    return law_dict


def identify_answer(law_dict, answer):
    propmt = f""" You are an expert in extracting a single best law related keyword from a query
    Find a single keyword from a 
    Query: {answer} based on {law_dict}
    If you can't find any keyword just emit NO!!
    """
    messages = [SystemMessage(content=propmt), HumanMessage(content="")]

    responses = model(messages)

    return responses


def search_dict_pattern(extraction, query):
    pattern = r"\{.*\}"
    match = re.search(pattern, extraction)
    if match:
        dictionary_str = match.group()
        try:
            extracted_dictionary = ast.literal_eval(dictionary_str)
            if isinstance(extracted_dictionary, set):
                return extracted_dictionary.pop()
            elif isinstance(extracted_dictionary, dict):
                if len(extracted_dictionary.values()) == 1:
                    return (
                        list(extracted_dictionary.keys())[0],
                        list(extracted_dictionary.values())[0][0],
                    )
                else:
                    value = select_value(extracted_dictionary, query)
                    return list(extracted_dictionary.keys())[0], value.content
            elif isinstance(extracted_dictionary, tuple):
                if len(extracted_dictionary) > 2:
                    last_element = extracted_dictionary[-1]
                    if isinstance(last_element,dict):
                        return list(last_element.keys())[0], last_element[list(last_element.keys())[0]][0]
                    elif isinstance(last_element,set):
                        return next(iter(last_element))
                else:    
                    return extracted_dictionary[0], extracted_dictionary[1]
        except  (SyntaxError, ValueError):
             pattern = r'\{[^{}]*\}'
             matches = re.findall(pattern, dictionary_str)
             if matches:
                extracted_dictionary = ast.literal_eval(dictionary_str)
                if isinstance(extracted_dictionary, dict):
                    if len(extracted_dictionary.values()) == 1:
                        return (
                            list(extracted_dictionary.keys())[0],
                            list(extracted_dictionary.values())[0][0],
                            )
                    else:
                        value = select_value(extracted_dictionary, query)
                        return list(extracted_dictionary.keys())[0], value.content

    else:
        return extraction


def main():
    law_dict = create_lawyer_dict("keywords.html")
    query = " I have encountered an intellectual property issue with a colleague who has wrongfully claimed credit for my work. I am seeking legal assistance. Can you provide guidance or help in this matter?"

    while True:
        query = input("Enter your query regarding the Legally:Yours \n >>>input:")
        if "End" in query:
            break
        refine = outline_guided(law_dict, query).content
        # print(colored(refine,"yellow"))
        refine = refine.lower()
        if "not relevant" in refine:
            print(colored(refine, "red"))
        else:
            keywords = ["personal", "business", "business law", "personal law"]
            pattern = "|".join(re.escape(keyword) for keyword in keywords)
            what_law = create_agent_schema(keywords, query).content
            print(colored(what_law, "red"))
            matches = re.findall(pattern, what_law, flags=re.IGNORECASE)
            if matches:
                if "business" in matches[0]:
                    extracted_answer = refine_keywords(
                        law_dict["business-law"], query
                    ).content
                    search_dict_pattern(extracted_answer, query)
                if "personal" in matches[0]:
                    extracted_answer = refine_keywords(
                        law_dict["personal-law"], query
                    ).content
                    print(extracted_answer)
                    search_dict_pattern(extracted_answer, query)


if __name__ == "__main__":
    main()

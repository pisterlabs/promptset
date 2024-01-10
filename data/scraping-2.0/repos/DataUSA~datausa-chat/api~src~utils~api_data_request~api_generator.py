import openai
import os
import pandas as pd
import requests
import time
import json

from os import getenv
from dotenv import load_dotenv
from langchain.llms import OpenAI
from src.utils.table_selection.table_details import get_table_api_base, get_drilldown_levels, get_table_columns, get_table_schemas
from src.utils.preprocessors.text import *
from src.utils.api_data_request.similarity_search import *

load_dotenv()

# environment initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# variable initialization
OPENAI_KEY = getenv("OPENAI_KEY")
openai.api_key = OPENAI_KEY

TESSERACT_API = getenv("TESSERACT_API")
MONDRIAN_API = getenv('MONDRIAN_API')


def get_api_components_messages(table):
    message = (
        """
        You are an expert data scientist working with data organized in a multidimensional format, such as in OLAP cubes.
        You are given the following natural language query from a user and information of a cube that contains the data to answer this query.

        ---------------\n
        {natural_language_query}
        \n---------------\n

        Your goal is to identify the variables, measures and filters that are relevant and are needed in order to retrieve this data from the cube through an API.
        Respond in JSON format with your answer separated into the following fields: 
        \"variables\" which is a list of strings that contain the variables.\n
        \"measures\" which is a list of strings that contain the relevant measures.\n
        \"filters\" which is a list of strings that contain the filters in the form of 'variable = filtered_value'.\n

        Write your answer in markdown format.\n
        """
    )

    response_part = """
        {{
            "variables": "",
            "measures": "",
            "filters": ""
        }}
        """

    return (
        message +
        f"""
        The following JSON contains the information of the cube you can query, with its measures, dimensions and hierarchies:\n
        ---------------------\n
        {get_table_columns([table])}
        ---------------------\n

        in your answer, provide the following information:\n

        - <one to two sentence comment explaining why the chosen variables, measures and filters can answer the query>\n
        - <for each variable, measure and filter identified, comment double checking that they exist in the table with the specified name>\n
        - the markdown formatted like this:\n
        ```
        {response_part}
        ```
        Provide only the list of variables, measures and filters, and nothing else after.\n
        A few rules to take into consideration:\n
        - You cannot apply filters to different variables with the same parent dimension. Choose only one (the most relevant or most granular)\n
        - Assume the latest year to be 2023.\n
        - For cases where the query requires to filter by a certain range of years or months, please specify all of them separately.
        """
    )


def get_api_params_from_lm(natural_language_query, table = None, model="gpt-4", top_matches=False):
    """
    Identify relevant tables for answering a natural language query via LM
    """
    max_attempts = 5
    attempts = 0

    content = get_api_components_messages(table).format(
        natural_language_query = natural_language_query,
        )

    messages = []
    messages.append({
        "role": "user",
        "content": content
    })
    
    while attempts < max_attempts:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
                )
        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out (attempt {attempts + 1}): {e}")
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error (attempt {attempts + 1}): {e}")
        except openai.error.APIConnectionError as e:
            print(f"OpenAI API request failed to connect: {e}")
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}")
        else:
            break
        attempts += 1
        time.sleep(1)

    output_text = response['choices'][0]['message']['content']
    print("\nChatGPT response:", output_text)
    params = extract_text_from_markdown_triple_backticks(output_text)
    print("\nParameters:", params)

    variables = json.loads(params).get("variables")
    measures = json.loads(params).get("measures")
    cuts = json.loads(params).get("filters")

    return variables, measures, cuts


def cuts_processing(cuts, cube_name, drilldowns):
    updated_cuts = {}
    
    for i in range(len(cuts)):
        var = cuts[i].split('=')[0].strip()
        cut = cuts[i].split('=')[1].strip()

        var_levels = get_drilldown_levels(cube_name, var)
        if var == "Year" or var == "Month" or var == "Quarter" or var == "Month and Year":
            if var in updated_cuts:
                updated_cuts[var].append(cut)
            else:
                updated_cuts[var] = [cut]
        else:
            drilldown_id, drilldown_name, s = get_similar_content(cut, cube_name, var_levels)

            if drilldown_name != var:
                    drilldowns.remove(var)
                    if drilldown_name not in drilldowns:
                        drilldowns.append(drilldown_name)


            if drilldown_name in updated_cuts:
                updated_cuts[drilldown_name].append(drilldown_id)
            else:
                updated_cuts[drilldown_name] = [drilldown_id]
            
    api_params = '&' + '&'.join([f"{key}={','.join(values)}" for key, values in updated_cuts.items()])
    
    return api_params, drilldowns


def api_build(table, drilldowns, measures, cuts, limit = ""):
    base = get_table_api_base(table)
    
    for i in range(len(drilldowns)):
        drilldowns[i] = clean_string(drilldowns[i])

    for i in range(len(measures)):
        measures[i] = clean_string(measures[i].replace("Index", "Inde"))

    measures_str = "&measures=" + ','.join(measures)
    cuts_str, drilldowns = cuts_processing(cuts, table, drilldowns)
    drilldowns_str = "&drilldowns=" + ','.join(drilldowns)

    if base == "Mondrian": base = MONDRIAN_API
    else: base = TESSERACT_API + "cube=" + table
    
    url = base + drilldowns_str + measures_str + cuts_str

    return url


def api_request(url):
    try:
        r = requests.get(url)
        r.raise_for_status()

        if 'data' in r.json():
            json_data = r.json()['data']
            df = pd.DataFrame.from_dict(r.json()['data'])
            return json_data, df, ""

    except: 
        json_data = json.loads('{}')
        df = pd.DataFrame()
        return json_data, df, "No data found."
        #raise ValueError('Invalid API url:', url)

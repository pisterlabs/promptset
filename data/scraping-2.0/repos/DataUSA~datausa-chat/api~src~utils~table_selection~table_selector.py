import json
import openai
import time
from typing import List
from sentence_transformers import SentenceTransformer
from os import getenv
from dotenv import load_dotenv

from src.utils.table_selection.table_details import get_table_schemas, get_table_names
from src.utils.table_selection.table_database_search import get_similar_tables
from src.utils.few_shot_examples import get_few_shot_example_messages
#from src.utils.messages import get_assistant_message_from_openai
from src.utils.preprocessors.text import extract_text_from_markdown_triple_backticks

load_dotenv()

OPENAI_KEY = getenv("OPENAI_KEY")
openai.api_key = OPENAI_KEY

def _get_table_selection_message_with_descriptions(table_names: List[str] = None, top_matches = False):
    message = (
        """
        You are an expert data scientist.\n
        Return a JSON object with the relevant SQL tables to answer the following natural language query:\n
        ---------------\n
        {natural_language_query}
        \n---------------\n
        Respond in JSON format with your answer in a field named \"tables\" which is a list of strings.\n
        Respond with an empty list if you cannot identify any relevant tables.\n
        Write your answer in markdown format.\n
        """
    )

    if top_matches == True:

        return (
            message +
            f"""
            The following are the tables you can query:\n
            ---------------------\n
            {get_table_schemas(table_names)}
            ---------------------\n

            in your answer, provide the following information:\n
            
            - <one to two sentence comment explaining why the chosen table is relevant goes here>\n
            - <for each table identified, comment double checking the table is in the schema above along with what the first column in the table is or (none) if it doesn't exist>\n
            - the markdown formatted like this:\n
            ```\n
            <json of the tables>\n
            ```\n

            Provide only the list of related tables and nothing else after.
            """
        )
    
    else:
        return (
            message +
            f"""
            The following is a table that can be used to answer the natural language query, along with the definition of its enums:\n
            ---------------------\n
            {get_table_schemas(table_names)}
            ---------------------\n

            in your answer, provide the following information:\n
            
            - <comment double checking the table is the one above along with what the first column in the table is or (none) if it doesn't exist.>\n
            - <select the most relevant table that can be used to answer the query.>
            - the markdown formatted like this:\n
            ```\n
            <json of the most relevant table>\n
            ```\n

            Provide only the most relevant table and nothing else after.
            """
        )


def _get_table_selection_messages() -> List[str]:
    """
    
    """
    default_messages = []
    default_messages.extend(get_few_shot_example_messages(mode = "table_selection"))
    return default_messages


def get_relevant_tables_from_database(natural_language_query, embedding_model = 'multi-qa-MiniLM-L6-cos-v1', content_limit=1) -> List[str]:
    """
    Returns a list of the top k table names (matches the embedding vector of the NLQ with the stored vectors of each table)
    """
    #vector = get_embedding(natural_language_query, "text-embedding-ada-002")

    model = SentenceTransformer(embedding_model) # 384
    vector = model.encode([natural_language_query])

    results = get_similar_tables(vector, content_limit=content_limit)

    return list(results)


def get_relevant_tables_from_lm(natural_language_query, table_list = None, model="gpt-4", session_id=None, top_matches=False) -> List[str]:
    """
    Identify relevant tables for answering a natural language query via LM
    """
    max_attempts = 5
    attempts = 0

    content = _get_table_selection_message_with_descriptions(table_list, top_matches=top_matches).format(
        natural_language_query = natural_language_query,
    )

    messages = _get_table_selection_messages().copy()
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
    tables_json_str = extract_text_from_markdown_triple_backticks(output_text)
    print("\nTables:", tables_json_str)

    table_list = json.loads(tables_json_str).get("tables")

    return table_list


def request_tables_to_lm_from_db(natural_language_query, content_limit=3):
    """
    Extracts most similar tables from database using embeddings and similarity functions, and then lets the llm choose the most relevant one.
    """
    tables = get_relevant_tables_from_database(natural_language_query, content_limit=content_limit)
    gpt_selected_table = get_relevant_tables_from_lm(natural_language_query, table_list = tables, top_matches=True)

    return gpt_selected_table
import json
from typing import Any, List, Dict
import openai
import requests

import logging
logger = logging.getLogger(__name__)


def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """

    prompt = f"""
        Du bist ein Assistent, der die Informationen hier drueber nutzt, um Fragen zu beantworten. Beantworte die Fragen so ausgiebig wie moeglich mit den vorhandenen informationen. Wenn die Information im Text nicht vorhanden ist, oder du dir nicht sicher bist sage: 'Es tut mir leid, ich kenne die Antwort nicht.' {question}
    """
    return prompt

def call_chatgpt_api_user_promt_system_prompt(user_prompt: str, system_prompt: str = None, engine: str = "kai-gpt-16k-model") -> Dict[str, Any]:
    """
    Call chatgpt API with a user prompt and an optional system prompt.
    
    Parameters:
    - user_prompt (str): The user's question or input to ask the model.
    - system_prompt (str, optional): An optional system message to prepend before the user's input. Default is None.
    - engine (str, optional): The OpenAI engine to use for the API call. Default is "kai-gpt-16k-model".
    
    Returns:
    - Dict[str, Any]: The response from the GPT-3 API.
    
    Raises:
    - Exception: If there's any error during the API call.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=messages,
            max_tokens=8000,
            temperature=0.3,
        )
        return response
    except Exception as e:
        # Handle the exception as required, for now, just printing it
        logger.error(f"Error occurred: {e}")
        raise e


def call_chatgpt_api(user_question: str, chunks: List[str] = None, engine: str = "kai-gpt-16k-model") -> Dict[str, Any]:
    """
    Call chatgpt API with user's question and retrieved chunks.
    
    Parameters:
    - user_question (str): The user's question to ask the model.
    - chunks (List[str], optional): A list of context chunks to prepend before the user's question.
    
    Returns:
    - Dict[str, Any]: The response from the GPT-3 API.
    """
    
    messages = [{"role": "user", "content": chunk} for chunk in (chunks or [])]
    messages.append({"role": "user", "content": user_question})
    
    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )
        return response
    except Exception as e:
        # Handle the exception as required, for now, just printing it
        logger.error(f"Error occurred: {e}")
        raise e


def ask(user_question: str, bearer_token_db: str, server_ip: str, max_characters_extra_info = 16000, source: str = "vector") -> Dict[str, Any]:
    """
    Handles user questions, queries a database, and generates responses using ChatGPT.

    - Queries the database with the user's question.
    - Logs the user's question and the retrieved chunks.
    - Calls ChatGPT with the question and chunks to generate a response.
    - Logs and returns the first generated response.

    Parameters:
    - user_question (str): The user's input question.
    - bearer_token_db (str): Token for database authentication.
    - server_ip (str): IP address of the server.
    - max_characters_extra_info (int, optional): Maximum character limit for extra info. Defaults to 16000.
    - source (str, optional): Data source type. Defaults to "vector".

    Returns:
    - Dict[str, Any]: Contains the generated response in "choices"[0]["message"]["content"].
    """
    chunks = ""
    if source == "vector":
        # Get chunks from database.
        chunks = query_database(user_question, bearer_token_db, server_ip, max_characters_extra_info=max_characters_extra_info)
    else:
        keywords = ask_direct_search(user_question)
        logger.info(f">>>>>> The keywords for direct search are: {keywords}")
        chunks = search_jsonl("/home/azureuser/phat_sharepoint.jsonl", keywords, max_characters_extra_info=max_characters_extra_info)

    logger.info(f">>>>>> {source} User's questions: {user_question}")
    logger.info(f">>>>>> {source} Use {len(chunks)} chunks")
    if(len(chunks) == 0):
        return "Es konnten keine Informationen zu dieser Frage gefunden werden."
    
    response = call_chatgpt_api(apply_prompt_template(user_question), chunks)

    return response["choices"][0]["message"]["content"]

def ask_direct_search(user_question: str) -> str:
    """
    Handles user questions using ChatGPT for direct keyword extraction.

    - Logs the user's question.
    - Calls ChatGPT with the question and specific prompts to extract keywords.
    - Returns the extracted keywords or relevant information.

    Parameters:
    - user_question (str): The user's input question.

    Returns:
    - str: Contains the extracted keywords or information
    """
    logger.info(">>>>>> User's questions: %s", user_question)
    response = call_chatgpt_api(f"""{user_question} ---- 
        antworte mit ja oder nein (ein wort): wird oben nach einer person bei namen gefragt?""")["choices"][0]["message"]["content"]
    if "ja" in response.lower():
        return call_chatgpt_api(f"""{user_question} ----
            helfe mir aus der frage oben nur den namen zu schreiben""")["choices"][0]["message"]["content"]
    return call_chatgpt_api(f"""{user_question} ----
        schreibe nur ein wort oder woerter: was sind die wichtigsten woerter in diesem satz oben?""")["choices"][0]["message"]["content"]

def query_database(query_prompt: str, bearer_token: str, server_ip: str, max_characters_extra_info: int = 16000) -> List[str]:
    """
    Queries a vector database and retrieves relevant text chunks based on the user's input.

    Parameters:
    - query_prompt (str): The user's input question or prompt for querying.
    - bearer_token (str): Authentication token for the database.
    - server_ip (str): IP address of the database server.
    - max_characters_extra_info (int, optional): Max character limit for the combined chunks. Defaults to 16000.

    Returns:
    - List[str]: List of retrieved text chunks.

    Raises:
    - ValueError: If there's an error in the database response.
    """
    url = f"http://{server_ip}:8000/query"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }
    data = {"queries": [{"query": query_prompt, "top_k": 22}]}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()

        chunks = []
        char_counter = 0
        for result in result["results"]:
            for inner_result in result["results"]:
                innter_text = inner_result["text"]
                char_counter = char_counter + len(innter_text)
                if char_counter > max_characters_extra_info:
                    continue
                logger.info(f">>>>>> Add following info to question from Milvus: {innter_text}")
                chunks.append(innter_text)

        # process the result
        return chunks
    else:
        raise ValueError(f"Error: {response.status_code} : {response.content}")

def search_jsonl(file_path: str, search_text: str, max_characters_extra_info: int = 16000) -> List[str]:
    """
    Search for words in a .jsonl file and return matching entries.
    
    Parameters:
    - file_path (str): Path to the .jsonl file.
    - search_text (str): Words to search for, separated by spaces. Special characters 
                         (ä, ü, ö, ß) are replaced with (ae, ue, oe, ss).
    - max_characters_extra_info (int, optional): Maximum combined character limit for returned entries. 
                                                 Defaults to 16000.
    
    Returns:
    - List[str]: List of entries sorted by the number of search words matched in descending order. 
                 The total character count of the list will be below the 'max_characters_extra_info' threshold.
    """
    search_text = search_text.lower().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss')
    search_words = [word.strip() for word in search_text.replace(",", "").replace('"', '').split()]
    
    entries_with_counts = []

    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            entry_text_lower = entry.get('text', '').lower()

            # print(entry.get('id', ''))
            # if "Kai_Luenstaeden" in entry.get('id', ''):
            #     search_word = search_words[0]
            #     count = entry_text_lower.count(search_words[0])
            #     is_in = search_words[0] in entry_text_lower
            #     print(f">>>>>>>>><<<<<<<<<<<<< {entry.get('id', '')} count: {count} is in: {is_in}")

            match_count = sum(entry_text_lower.count(word) for word in search_words)
            if match_count:
                entries_with_counts.append((entry, match_count))
    
    sorted_texts = [entry["text"] for entry, _ in sorted(entries_with_counts, key=lambda x: x[1], reverse=True)]
    
    final_texts = []
    char_counter = 0
    for text in sorted_texts:
        char_counter += len(text)
        if char_counter > max_characters_extra_info:
            break
        final_texts.append(text)
        logger.info(f">>>>>> Add following info to question from Direct Search: {text}")
                
    return final_texts
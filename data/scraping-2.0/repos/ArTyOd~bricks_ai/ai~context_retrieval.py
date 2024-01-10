from .llm_interactions import answer_question

import re
import json

from typing import List, Dict, Union, Tuple, Optional

from urllib.parse import urlparse, parse_qs
from typing import Union, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
import time


def extract_context(text: str) -> Dict[str, Optional[str]]:
    """
    Extract context information from a given text.

    This function identifies keys like 'Fahrzeug-Flag', 'Fahrzeug', and 'Felge' and extracts
    their corresponding values. It also looks for URLs and parses their query parameters.

    Parameters:
        text (str): The input text from which context information is extracted.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing the extracted context information.
        Keys are renamed to 'vehicle_flag', 'vehicle', and 'rim'.
    """
    context = {}
    key_rename_map = {
        "Fahrzeug-Flag": "vehicle_flag",
        "Fahrzeug": "vehicle",
        "Felge": "rim",
    }

    try:
        pre_context, system_context = text.split("Fahrzeug-Flag:", 1)
        context["question"] = pre_context.strip()
        system_context = f"Fahrzeug-Flag:{system_context}"  # Add "Fahrzeug-Flag:" back to system_context

        keys = list(key_rename_map.keys())
        for i in range(len(keys)):
            old_key = keys[i]
            new_key = key_rename_map[old_key]

            if i < len(keys) - 1:
                next_old_key = keys[i + 1]
                match = re.search(f"{old_key}:(.*?){next_old_key}:", system_context, re.DOTALL)
            else:
                match = re.search(f"{old_key}:(.*)", system_context, re.DOTALL)

            if match:
                value = match.group(1).strip()
                context[new_key] = value

        if context.get("rim"):
            try:
                _, link = system_context.split("https://", 1)
                full_link = f"https://{link.strip()}"
                parsed_url = urlparse(full_link)
                query_params = parse_qs(parsed_url.query)
                context["possible_rim_id"] = query_params.get("possible_rim_id", [None])[0]
                context["rim_entry_id"] = query_params.get("rim_entry_id", [None])[0]
            except ValueError:
                pass
    except ValueError:
        context["pre_context"] = text

    return context


def extract_rim_details_from_response(response: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract rim details from a GPT response.

    Parameters:
    - response (str): The response string from GPT containing rim details.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries where each dictionary contains the rim details.
    """
    felgen = [f.strip() for f in re.split(r"Felge \d+:", response) if f.strip()]

    properties = [
        "hersteller",
        "design",
        "farbe",
        "lochkreise",
        "zoll",
        "breite",
        "einpresstiefe",
        "nabe",
        "traglast",
        "gewicht",
    ]

    details_list = [{prop: extract_property(felge, prop) for prop in properties} for felge in felgen]

    return details_list


def extract_property(text: str, property_name: str) -> Optional[str]:
    """
    Extract a specific property value from a text based on the property name.

    Parameters:
    - text (str): The text from which the property value needs to be extracted.
    - property_name (str): The name of the property whose value needs to be extracted.

    Returns:
    - str or None: The extracted value of the property or None if not found.
    """
    match = re.search(f"{property_name}: (.+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def get_context_for_rim_extraction(row: Dict[str, Union[str, None]], json_data: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    """
    Determine the context and instruction for extraction based on the row data.

    Parameters:
    - row (Dict[str, Union[str, None]]): A dictionary containing the data of a row (e.g., from a DataFrame).
    - json_data (Dict[str, str]): A dictionary loaded from a JSON file containing instructions.

    Returns:
    - Tuple[Dict[str, str], str]: A tuple containing the context dictionary and the instruction string.
    """
    if row.get("rim") is None:
        return {"question": f"{row['question']}"}, json_data["extract_rim_details"]
    else:
        return {
            "question": f"{row['question']}",
            "rim details": f"{row['rim']}",
        }, json_data["extract_rim_details_with_rim_details"]


def get_rim_details_for_question(question: str, instruction: str, debug: bool = False) -> List[Dict[str, Optional[str]]]:
    """
    Extract rim details for a given question using a specific instruction.

    Parameters:
    - question (str): The question based on which the rim details need to be extracted.
    - instruction (str): The instruction to guide the extraction.

    Returns:
    - List[Dict[str, Optional[str]]]: A list of dictionaries containing the extracted rim details.
    """
    prompt = [{"role": "user", "content": f"{question}"}]

    # Placeholder for `answer_question` function
    response, _ = answer_question(model="gpt-4", instruction=instruction, prompt=prompt, debug=debug)
    print(f"\n {response =} \n")
    return extract_rim_details_from_response(response)


def map_similar_rims1(
    question: str,
    df_all_rims_or_path: Union[pd.DataFrame, str],
    max_len: int = 1200,
    engine_name: str = "text-embedding-ada-002",
    emb_col_name: str = "emb hersteller design",
    context_fields: List[str] = ["hersteller", "design"],
    detail_fields: List[str] = ["similarity"],
    limit: int = 5,
    threshold: float = 0.95,
) -> Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, float]]]:
    """
    Create a context for a question by finding the most similar rims from a DataFrame.

    Parameters:
        question (str): The question to map.
        df_all_rims_or_path (Union[pd.DataFrame, str]): DataFrame or path to the DataFrame containing rim details.
        max_len (int): Maximum length of the context. Default is 1200.
        engine_name (str): Name of the text embedding engine. Default is "text-embedding-ada-002".
        emb_col_name (str): Column name for embeddings in DataFrame. Default is "emb hersteller design".
        context_fields (List[str]): Fields to include in the returned context. Default is ['hersteller', 'design'].
        detail_fields (List[str]): Fields to include in the returned details. Default is ['similarity'].
        limit (int): Maximum number of contexts to return. Default is 5.
        threshold (float): Minimum similarity threshold to include a context. Default is 0.95.

    Returns:
        Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, float]]]: Returns contexts and their details.
    """

    # Initialize DataFrame
    if isinstance(df_all_rims_or_path, pd.DataFrame):
        df_all_rims = df_all_rims_or_path
    else:
        df_all_rims = pd.read_pickle(df_all_rims_or_path)

    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine=engine_name)["data"][0]["embedding"]

    # Prepare embeddings for comparison
    embeddings = np.array(df_all_rims[emb_col_name].tolist())

    # Calculate similarities
    similarities = cosine_similarity([q_embed], embeddings)[0]

    # Add similarities to DataFrame
    df_all_rims["similarity"] = similarities

    # Sort by similarity
    sorted_df = df_all_rims.sort_values(by="similarity", ascending=False)

    contexts, context_details = [], []

    for _, row in sorted_df.iterrows():
        # Check for limit and threshold
        if len(contexts) >= limit or row["similarity"] < threshold:
            break

        context_data = {field: row[field] for field in context_fields}
        contexts.append(context_data)

        detail_data = {field: row[field] for field in detail_fields}
        context_details.append(detail_data)

    return contexts


def map_similar_rims(
    question: str,
    index,
    filter_query,
    engine_name: str = "text-embedding-ada-002",
    limit: int = 5,
    threshold: float = 0.95,
    max_len: int = 1200,
    context_fields: List[str] = ["hersteller", "design"],
    detail_fields: List[str] = ["similarity"],
) -> Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, float]]]:
    """
    Create a context for a question by finding the most similar rims from a Pinecone index.

    Parameters:
        question (str): The question to map.
        index: Pinecone index.
        ...

    Returns:
        Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, float]]]: Returns contexts and their details.
    """
    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine=engine_name)["data"][0]["embedding"]

    # Query Pinecone index
    res = index.query(q_embed, filter=filter_query, top_k=limit, include_metadata=True)
    # res = index.query(q_embed, top_k=limit, include_metadata=True)

    contexts, context_details = [], []

    for match in sorted(res["matches"], key=lambda x: x["score"], reverse=True):
        # Check for threshold
        if match["score"] < threshold:
            break

        # Extract metadata
        metadata = match["metadata"]

        context_data = {field: metadata.get(field, "") for field in context_fields}
        contexts.append(context_data)

        detail_data = {field: match["score"] for field in detail_fields}
        context_details.append(detail_data)

    return contexts


from typing import Union, List, Dict, Optional
import pandas as pd


def extract_manufacturer_and_design(row: Dict[str, Union[str, float]]) -> Optional[str]:
    """
    Extract and format manufacturer and design details from a row dictionary.

    Parameters: row (Dict[str, Union[str, float]]): The row containing the details.

    Returns:
        Optional[str]: The formatted string containing the manufacturer and design, or None if an error occurs.
    """
    try:
        return f"Hersteller: {row.get('hersteller', '')}; Design: {row.get('design', '')}"
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_vehicle_related_details(row: Dict[str, Union[str, float]]) -> Optional[str]:
    """
    Extract and format vehicle-related details from a row dictionary.

    Parameters:
        row (Dict[str, Union[str, float]]): The row containing the details.

    Returns:
        Optional[str]: The formatted string containing the vehicle-related details, or None if an error occurs.
    """
    try:
        result = f"Lochkreise: {row.get('lochkreise', '')}; Einpresstiefe: {row.get('einpresstiefe', '')}; Nabe: {row.get('nabe', '')}"
        print(f"extract_vehicle_related_details: {result =}")
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def safe_convert_to_float(val: Union[str, float, int]) -> Union[float, str]:
    """
    Safely convert a value to float.
    If conversion fails, return the original value.

    Parameters:
        val (Union[str, float, int]): The value to be converted.

    Returns:
        Union[float, str]: The converted float or the original value.
    """
    try:
        return float(val)
    except ValueError:
        return val


def convert_df_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert specific columns of a DataFrame to float and then to string.
    Replace 'None' with np.nan for the entire DataFrame.

    Parameters: df (pd.DataFrame): The DataFrame to be converted.

    Returns: pd.DataFrame: The converted DataFrame.
    """
    # Replace all 'None' values with np.nan for the entire dataframe
    df = df.replace("None", np.nan)

    # Columns to be safely converted to float
    columns_to_float = ["zoll", "breite", "einpresstiefe", "nabe"]

    for col in columns_to_float:
        if col in df.columns:
            df[col] = df[col].apply(safe_convert_to_float)

    # Columns to be converted to string
    columns_to_str = [
        "vehicle_flag",
        "vehicle",
        "hersteller",
        "design",
        "farbe",
        "lochkreise",
        "zoll",
        "breite",
        "einpresstiefe",
        "nabe",
        "traglast",
    ]

    for col in columns_to_str:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def filter_dataframe(
    row: pd.Series,
    columns_to_match: List[str],
    limit: int,
    rims_df_or_path: Union[pd.DataFrame, str] = "data/fo/df_all_rims.pkl",
) -> pd.DataFrame:
    """
    Filter a DataFrame based on the hersteller_design_mapped list.

    Parameters:
        row (pd.Series): The row containing filtering criteria.
        columns_to_match (List[str]): The columns to consider for matching.
        limit (int): The limit for hersteller_design_mapped list.
        rims_df_or_path (Union[pd.DataFrame, str]): The DataFrame or path to the DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    if isinstance(rims_df_or_path, pd.DataFrame):
        df_all_rims = rims_df_or_path
    else:
        df_all_rims = pd.read_pickle(rims_df_or_path)

    row.to_pickle("data/test_row.pkl")
    hersteller_design_mapped = row["hersteller_design_mapped"][limit:]
    filtered_df = pd.DataFrame()

    for context_item in hersteller_design_mapped:
        hersteller_value = context_item["hersteller"]
        design_value = context_item["design"]

        temp_df = df_all_rims[(df_all_rims["hersteller"] == hersteller_value) & (df_all_rims["design"] == design_value)]
        filtered_df = pd.concat([filtered_df, temp_df])

        for column in columns_to_match:
            value_to_match = row[column]
            if value_to_match != "nan":
                temp_filtered = filtered_df[filtered_df[column] == value_to_match]
                if not temp_filtered.empty:
                    filtered_df = temp_filtered

    return filtered_df


def get_mapping_information(vehicle_flag: str, rim_entry_id: str) -> Union[Dict[str, Any], str]:
    """
    Fetch mapping information from an external service.

    Parameters:
        vehicle_flag (str): The vehicle flag to be passed as a parameter.
        rim_entry_id (str): The rim entry ID to be passed as a parameter.

    Returns:
        Union[Dict[str, Any], str]: The mapping information or an error message.
    """
    url = "https://www.felgenoutlet.de/frontend/desktop/ki/mapping_information"
    params = {"vehicle_flag": vehicle_flag, "rim_entry_id": rim_entry_id}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


def fetch_and_update(row: pd.Series) -> Union[Dict[str, Any], None]:
    """
    Fetch and update information based on the vehicle flag and rim details.

    Parameters:
        row (pd.Series): The row containing the vehicle flag and rim details.

    Returns:
        Union[Dict[str, Any], None]: The fetched mapping information or None.
    """

    vehicle_flag = row.get("vehicle_flag", pd.NA)
    rim_details_systems = row.get("rim_details_system", [])
    if pd.notna(vehicle_flag) and rim_details_systems:
        for rim_detail in rim_details_systems:
            rim_entry_id = rim_detail.get("felgen_anwendungs_id", pd.NA)
            if pd.notna(rim_entry_id):
                mapping_info = get_mapping_information(vehicle_flag, rim_entry_id)
                time.sleep(13)
                if mapping_info != {"error": "Combination not found"}:
                    return mapping_info
    return None


def create_filter_query(hd: List[Dict[str, str]], zoll: Optional[str] = None) -> Dict:
    """
    Create a MongoDB filter query based on the input list of dictionaries containing 'hersteller', 'design', and 'zoll'.

    Parameters:
        hd (List[Dict[str, str]]): A list of dictionaries, each containing 'hersteller' and 'design' fields.
        zoll (Optional[str]): The zoll value as a string. Default is None.

    Returns:
        Dict: A MongoDB filter query.
    """

    filter_query = {"$and": [{"category": "vehicle rim"}]}

    for item in hd:
        hersteller_value = item.get("hersteller", "")
        design_value = item.get("design", "")

        if hersteller_value:
            filter_query["$and"].append({"hersteller": hersteller_value})

        if design_value:
            filter_query["$and"].append({"design": design_value})

    if zoll:
        filter_query["$and"].append({"zoll": zoll})

    return filter_query


def filter_rim_details(columns_to_match: List[str], row: Dict[str, any]) -> List[Dict[str, any]]:
    """
    Filter a list of rim details based on specified columns and values.

    Parameters:
        columns_to_match (List[str]): List of columns to filter by.
        row (Dict[str, Any]): Dictionary containing values to match for each specified column.

    Returns:
        List[Dict[str, Any]]: Filtered list of dictionaries containing rim details.
    """
    # Convert the list of dictionaries to a DataFrame
    rim_details = row["rim_details_system"]

    df = pd.DataFrame(rim_details)

    # Iterate through the specified columns to apply filtering
    for column in columns_to_match:
        value_to_match = row.get(column, "nan")

        if value_to_match != "nan":
            temp_filtered = df[df[column].str.contains(str(value_to_match), case=False, na=False)]

            if not temp_filtered.empty:
                df = temp_filtered

    # Convert the filtered DataFrame back to a list of dictionaries
    return df.to_dict("records")


def context_retrieval_entry(
    text: str,
    df_requests: pd.DataFrame,
    df_rim_details: pd.DataFrame,
    index,
    instruction: Optional[Dict] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Create temporary dataframes with the same columns as the original dataframes
    temp_df_requests = pd.DataFrame(columns=df_requests.columns)
    temp_df_rim_details = pd.DataFrame(columns=df_rim_details.columns)

    request = extract_context(text)

    temp_df_requests = pd.concat([temp_df_requests, pd.DataFrame([request])], ignore_index=True)

    context, updated_instruction = get_context_for_rim_extraction(request, instruction)
    results = get_rim_details_for_question(context, updated_instruction, debug=debug)
    row = temp_df_requests.iloc[-1]
    base_dict = {
        "requests_index": row.name,
        "vehicle_flag": row["vehicle_flag"],
        "vehicle": row["vehicle"],
    }

    # Use a list comprehension to combine the base_dict with each dictionary in results
    combined_data = [{**base_dict, **res} for res in results]
    # print(f"\n{combined_data = } \n")
    temp_df_rim_details = pd.concat([temp_df_rim_details, pd.DataFrame(combined_data)], ignore_index=True)
    temp_df_rim_details["hersteller_design_mapped"] = temp_df_rim_details.apply(
        lambda row: map_similar_rims(
            question=extract_manufacturer_and_design(row),
            index=index,
            filter_query={"category": "hersteller design"},
            engine_name="text-embedding-ada-002",  # You can replace this with your own engine name
            max_len=1200,
            limit=1,
            threshold=0.85,
            context_fields=["hersteller", "design"],
            detail_fields=["similarity"],
        ),
        axis=1,
    )
    # print(f"1 :{temp_df_rim_details['zoll']}")
    temp_df_rim_details = convert_df_to_str(temp_df_rim_details)
    # print(f"2: {temp_df_rim_details['zoll']}")
    # Store the range of indices that will be added to df_rim_details for this row
    start_index = len(df_rim_details)
    end_index = start_index + len(results) - 1
    temp_df_requests["rim_details_index"] = temp_df_requests["rim_details_index"].astype("object")
    temp_df_requests.at[0, "rim_details_index"] = list(range(start_index, end_index + 1))

    context_fields = [
        "felgen_id",
        "felgen_anwendungs_id",
        "hersteller",
        "design",
        "farbe",
        "lochkreise",
        "zoll",
        "breite",
        "einpresstiefe",
        "nabe",
        "traglast",
        "gewicht",
    ]
    columns_to_match = [
        "farbe",
        "lochkreise",
        "zoll",
        "breite",
        "einpresstiefe",
        "nabe",
    ]

    temp_df_rim_details["rim_details_system"] = temp_df_rim_details.apply(
        lambda row: map_similar_rims(
            question=extract_vehicle_related_details(row),
            index=index,
            filter_query=create_filter_query(row["hersteller_design_mapped"], row["zoll"]),
            engine_name="text-embedding-ada-002",  # You can replace this with your own engine name
            max_len=1200,
            limit=30,
            threshold=0.85,
            context_fields=context_fields,
        ),
        axis=1,
    )

    temp_df_rim_details["rim_details_system"] = temp_df_rim_details.apply(lambda row: filter_rim_details(columns_to_match, row), axis=1)

    temp_df_rim_details["mapping_information"] = temp_df_rim_details.apply(lambda row: fetch_and_update(row), axis=1)

    return temp_df_requests, temp_df_rim_details

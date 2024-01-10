"""Utilities for the language model."""
import ast
import json
import os
import pickle
import re
from typing import Any, Dict, List

import pandas as pd
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2TokenizerFast

# Load the configuration
cfg = OmegaConf.load("conf/config.yaml")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text))


def write_jsonl(data: Dict[str, Any], filepath: str) -> None:
    """Write the data to a JSONL file."""
    with open(filepath, "w") as file:
        for key, value in data.items():
            # Create a new dictionary with the key included
            entry = {key: value}
            file.write(json.dumps(entry) + "\n")


def index_pdf(pdf_file: str, cfg: DictConfig, p: Any, write: bool = False) -> dict:
    """Index a PDF file using gorbin."""
    # check if pdf_file exist
    if not os.path.exists(pdf_file):
        raise ValueError(f"{pdf_file} does not exist")

    logger.info(f"crunching {pdf_file} through gorbin ...")
    logger.info(f"my glob pattern:{pdf_file.split('/')[-1].replace('.pdf', '')}*")
    logger.info(f"my filepath: {cfg.data.pdf_path}")
    loader = GenericLoader.from_filesystem(
        cfg.data.pdf_path + "/",
        glob=f"{pdf_file.split('/')[-1].replace('.pdf', '')}*",
        suffixes=[".pdf"],
        parser=GrobidParser(segment_sentences=True),
    )
    documents = loader.load()

    logger.info(f"parsed {len(documents)} docs using gorbin")

    data = []
    unique_texts = set()

    if len(documents) == 0:
        logger.error(f"Failed to grobid file: {pdf_file}")

    for t in documents:
        text = t.page_content
        file_path = t.metadata["file_path"]

        unique_key = (text, file_path)

        if unique_key not in unique_texts:
            unique_texts.add(unique_key)
            d = {}
            d["text"] = text
            d["page"] = ast.literal_eval(t.metadata["pages"])[
                0
            ]  # take only the first page
            d["section"] = t.metadata["section_title"]
            d["paper"] = t.metadata["paper_title"]
            d["file"] = file_path
            data.append(d)

    filename = t.metadata["file_path"]
    df = pd.DataFrame(data).drop(columns=["paper", "file"])

    # data_section: Dict[str, Union[List[str], str]] = {}
    # data_page: Dict[Any, Union[List[str], str]] = {}

    data_section: Dict[Any, Any] = {}
    data_page: Dict[Any, Any] = {}

    unique_texts = set()

    # label each identified sentence
    for t in documents:
        text = t.page_content
        file_path = t.metadata["file_path"]
        raw_section = t.metadata["section_title"]
        section = clean_section_names(raw_section, p)
        page_number = ast.literal_eval(t.metadata["pages"])[
            0
        ]  # take only the first page

        unique_json_key = (text, file_path, section)

        if unique_json_key not in unique_texts:
            unique_texts.add(unique_json_key)  # type: ignore[arg-type]

            # Check if the section already exists in the data dictionary
            if section not in data_section:
                data_section[section] = []
            data_section[section].append(text)

            if page_number not in data_page:
                data_page[page_number] = []
            data_page[page_number].append(text)

    # Concatenating texts within each section
    for section, texts in data_section.items():
        data_section[section] = " ".join(texts)

    for page_number, texts in data_page.items():
        data_page[page_number] = " ".join(texts)

    if not os.path.exists(cfg.data.pdf_preprocessed_path):
        os.makedirs(cfg.data.pdf_preprocessed_path)

    filename = file_path.replace(
        cfg.data.pdf_path, cfg.data.pdf_preprocessed_path
    ).replace(".pdf", ".pkl")

    # Combine the variables into a dictionary
    data_to_save = {
        "data_section": data_section,
        "data_page": data_page,
        "dataframe": df,
    }

    if write:
        # Writing the combined data to a pickle file
        with open(filename, "wb") as file:
            pickle.dump(data_to_save, file)

        write_jsonl(data_section, filename.replace(".pkl", "_section.jsonl"))
        write_jsonl(data_page, filename.replace(".pkl", "_page.jsonl"))

        logger.info(f"Data saved to {filename}")

    return data_to_save


def read_jsonl_to_dict(filepath: str) -> dict:
    """Reads a JSONL file and returns a dictionary."""
    data_dict = {}
    with open(filepath, "r") as file:
        for line in file:
            # Convert JSON string to dictionary
            json_obj = json.loads(line.strip())
            # Extract key and value
            if json_obj:
                key, value = next(iter(json_obj.items()))
                data_dict[key] = value
            else:
                print("Warning: Empty JSON object. Skipping this line.")
    return data_dict


def clean_section_names(key: str, p: Any) -> str:
    """Clean dict key to remove special characters."""
    # Remove '.', ',', and '-' using regex
    cleaned_key = re.sub(r"[.,-]", "", key)
    # Strip leading/trailing whitespace and convert to lowercase
    cleaned_key = cleaned_key.strip().lower()

    # Split the key into words
    words = cleaned_key.split()

    # Apply singularization if there are less than three words
    if len(words) < 4:  # material and methods
        singularized_words = [
            p.singular_noun(word) if p.singular_noun(word) else word for word in words
        ]
        return " ".join(singularized_words)

    return cleaned_key


def filter_by_cumulative_length(
    examples: List, example_token_counts: List[int], threshold: int
) -> List:
    """Filter a list of examples by cumulative length."""
    paired_examples = sorted(zip(examples, example_token_counts), key=lambda x: x[1])

    # Extract the sorted examples and their corresponding token counts
    sorted_examples, sorted_token_counts = zip(*paired_examples)

    filtered_examples = []
    accepted_examples = []
    current_length = 0

    for example, count in zip(sorted_examples, sorted_token_counts):
        if current_length + count <= threshold:
            filtered_examples.append(example)
            accepted_examples.append(count)
            current_length += count
        else:
            break  # Stop adding more examples once the threshold is reached or exceeded

    return filtered_examples


def exclude_section_and_after(
    pdf_sections_dict: dict, exclude_and_after_list: List[str]
) -> dict:
    """Exclude the section and after from the pdf_sections_dict."""
    keys_to_remove = []
    found_exclude_part = False
    for key in pdf_sections_dict.keys():
        if found_exclude_part:
            keys_to_remove.append(key)
            continue

        for exclude_term in exclude_and_after_list:
            if exclude_term == key:  # found the exact exclude term
                keys_to_remove.append(key)
                found_exclude_part = True
                break

    # Step 2: Remove the keys
    for key in keys_to_remove:
        del pdf_sections_dict[key]
    return pdf_sections_dict


def exclude_section_and_before(
    pdf_sections_dict: dict, exclude_and_before_list: List[str]
) -> dict:
    """Exclude the section and before from the pdf_sections_dict."""
    keys_to_remove = set()
    # Step 1: Identify keys for removal
    include_next_keys = False
    for key in reversed(pdf_sections_dict.keys()):
        if key in exclude_and_before_list or include_next_keys:
            keys_to_remove.add(key)
            include_next_keys = True

    # Step 2: Remove the keys
    for key in keys_to_remove:
        del pdf_sections_dict[key]

    return pdf_sections_dict


def exclude_terms(pdf_sections_dict: dict, exclude_list: list[str]) -> dict:
    """Exclude the section and after from the pdf_sections_dict."""
    # Step 1: Identify keys for removal
    keys_to_remove = [
        key
        for key in pdf_sections_dict
        if any(exclude_term in key for exclude_term in exclude_list)
    ]

    # Step 2: Remove the keys
    for key in keys_to_remove:
        del pdf_sections_dict[key]
    return pdf_sections_dict


def old_extract_material_method(pdf_sections_dict: dict) -> dict:
    """Extract the material and method from the pdf_sections_dict."""
    exclude_and_after_list = ["conclusion", "result"]
    exclude_and_before_list = ["introduction", "abstract"]
    # 31708475, 33406425 has early discussion but 20213684 has late discussion
    exclude_list = ["ethic", "discussion", "declaration of interes", "statistical"]
    priority_section = ["material and method"]

    relevant_keys = [key for key in pdf_sections_dict if key in priority_section]
    logger.info(f"relevant_keys: {relevant_keys}")
    if relevant_keys:
        filtered_dict = {key: pdf_sections_dict[key] for key in relevant_keys}
        return filtered_dict

    else:
        logger.info("going through list of exclusion term")
        pdf_sections_dict_processed = exclude_section_and_after(
            pdf_sections_dict, exclude_and_after_list
        )
        pdf_sections_dict_processed = exclude_section_and_before(
            pdf_sections_dict_processed, exclude_and_before_list
        )
        pdf_sections_dict_processed = exclude_terms(
            pdf_sections_dict_processed, exclude_list
        )
        return pdf_sections_dict_processed


def format_dict_to_string(your_dict: dict) -> str:
    """Format the dict to string for prompt."""
    return "\n".join([f"{key}: {value}" for key, value in your_dict.items()])


def format_dataframe_to_string(df: pd.DataFrame) -> str:
    """Formats each row of the DataFrame into a specified string format."""
    formatted_str = [
        f"content of section {row['section']} is: {row['text']}\n"
        for _, row in df.iterrows()
    ]
    return "\n".join(formatted_str)


def return_labelled_data(cfg: DictConfig, type: str) -> pd.DataFrame:
    """Return the labelled data as a dataframe."""
    df = pd.read_csv(cfg.data.labeled_rag_path)
    logger.info(f"reading {len(df)} labeled data")
    if type not in list(df["key"].unique()):
        raise ValueError(f"{type} must be {set(df['key'].unique())}")

    # filter the dataframe to only include the device data
    df_device = df.dropna(subset=["context"]).query(f"key == '{type}' & value != 'Nan'")
    logger.info(f"found {len(df_device)} labeled data for {type}")

    return df_device

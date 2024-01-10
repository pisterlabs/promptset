"""Scripts to run QAOverDocs Experiments
"""
import argparse
import json
import os
from datetime import datetime
from importlib import import_module
from shutil import copyfile

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import ARTIFACT_DIR, DATA_DIR, DOCUMENT_SOURCE, EXCLUDE_DICT, LOGGER
from exp import MapReduceQAOverDocsExperiment, RefineQAOverDocsExperiment
from utils import convert_csv_to_documents, convert_json_to_documents, load_documents


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Run Experiment")

    # Project Level Arguments
    parser.add_argument("--yaml_cfg", "-y", type=str, help="path_to_yaml_file")

    args = parser.parse_args()
    return args


def main():
    """Main Execution Function"""
    args = get_argument_parser()

    with open(args.yaml_cfg, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    project = yaml_cfg["project"]
    prompt_file = yaml_cfg["prompt"]
    test_case_path = yaml_cfg["test_case"]
    document_level = yaml_cfg["document_level"]
    chunk_size = yaml_cfg["chunk_size"]
    chunk_overlap = yaml_cfg["chunk_overlap"]
    additional_docs = yaml_cfg["additional_docs"]
    description = yaml_cfg["description"]
    chain_type = yaml_cfg["chain_type"]
    ground_truth = yaml_cfg["ground_truth"]

    if chain_type == "map_reduce":
        map_prompt = import_module(
            f"prompts.{project}.{os.path.splitext(prompt_file)[0]}"
        ).CHAT_QUESTION_PROMPT

        combine_prompt = import_module(
            f"prompts.{project}.{os.path.splitext(prompt_file)[0]}"
        ).CHAT_COMBINE_PROMPT

        try:
            collapse_prompt = import_module(
                f"prompts.{project}.{os.path.splitext(prompt_file)[0]}"
            ).CHAT_COLLAPSE_PROMPT

        except ImportError:
            collapse_prompt = None

        experiment = MapReduceQAOverDocsExperiment(
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            collapse_prompt=collapse_prompt,
            gt=os.path.join(DATA_DIR, "queries", ground_truth),
            **yaml_cfg,
        )

    elif chain_type == "refine":
        initial_prompt = import_module(
            f"prompts.{project}.{os.path.splitext(prompt_file)[0]}"
        ).CHAT_INITIAL_PROMPT

        refine_prompt = import_module(
            f"prompts.{project}.{os.path.splitext(prompt_file)[0]}"
        ).CHAT_REFINE_PROMPT

        experiment = RefineQAOverDocsExperiment(
            initial_prompt=initial_prompt,
            refine_prompt=refine_prompt,
            gt=os.path.join(DATA_DIR, "queries", ground_truth),
            **yaml_cfg,
        )

    LOGGER.info(
        "Successfully created experiment with settings:\n{}".format(
            "\n".join([f"{k}:{v}" for k, v in yaml_cfg.items()])
        )
    )

    with open(test_case_path, "r") as f:
        test_cases = f.readlines()

    documents = load_documents(
        os.path.join(DOCUMENT_SOURCE, project), exclude_pages=EXCLUDE_DICT
    )

    if document_level == "chunk":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)

    if additional_docs:
        with open(additional_docs, "r") as f:
            add_doc_infos = json.load(f)
        for add_doc_info in add_doc_infos:
            if add_doc_info["mode"] == "table":
                documents.extend(
                    convert_csv_to_documents(add_doc_info, concatenate_rows=True)
                )
            elif add_doc_info["mode"] == "json":
                documents.extend(convert_json_to_documents(add_doc_info))
            else:
                LOGGER.warning(
                    "Invalid document type. No texts added to documents list"
                )

    experiment.run_test_cases(
        test_cases, docs=documents, return_intermediate_steps=True
    )

    LOGGER.info("Completed running all test cases.")

    save_path = os.path.join(
        ARTIFACT_DIR,
        "{}_{}".format(description, datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )

    os.makedirs(save_path, exist_ok=True)
    experiment.save_json(os.path.join(save_path, "result.json"))
    experiment.write_csv(os.path.join(save_path, "result.csv"))

    copyfile(args.yaml_cfg, os.path.join(save_path, "settings.yaml"))


if __name__ == "__main__":
    main()

# python3 ./src/scripts/qa_over_docs_experiment.py --yaml_cfg exps/map_reduce_1.yaml

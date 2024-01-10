import logging
import os

import openai
import pandas as pd
from dotenv import load_dotenv

from claims_analysis.constants import THREADS, ExtendedCoverage
from claims_analysis.page_processing import Violation, process_claim_pages
from claims_analysis.summarization import ClaimSummary, summarize_results
from claims_analysis.utils import convert_pdf_to_page_list, log_timer, setup_logging

CLAIMS_DIR = None
OUTPUTS_DIR = None
LOGS_DIR = None


@log_timer
def __configure_file_paths(is_cloud_run: bool, config_data_parameters: dict):
    """Configure OPENAI_API_KEY, Claims, Outputs and Logs folder paths

    Args:
        is_cloud_run (bool): True in case the execution started from the Google Colab Notebook, otherwise False.
        config_data_parameters (dict): A dictionary containing OPENAI_API_KEY, Claims, Outputs and Logs folder paths.
    """
    global CLAIMS_DIR, OUTPUTS_DIR, LOGS_DIR

    if is_cloud_run:

        # Setup API key passed as a parameter from colab notebook
        os.environ["OPENAI_API_KEY"] = config_data_parameters["OPENAI_API_KEY"]

        # Setup Claims, Outputs and Logs file paths passed as a parameter from colab notebook
        CLAIMS_DIR = config_data_parameters["CLAIMS_DIR"]
        OUTPUTS_DIR = config_data_parameters["OUTPUTS_DIR"]
        LOGS_DIR = config_data_parameters["LOGS_DIR"]

    else:
        # Setup API key locally
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Setup Claims, Outputs and Logs file paths to access locally
        CLAIMS_DIR = "claims/"
        OUTPUTS_DIR = "outputs/"
        LOGS_DIR = "logs/"


@log_timer
def process_single_claim(
    claim_path: str, extended_coverages: list[ExtendedCoverage] = []
) -> tuple[list[Violation], ClaimSummary]:
    """Read claim, get violations, and summarization for a single claim."""

    # Read the claim
    pages = convert_pdf_to_page_list(claim_path)

    # Get all violations and the page numbers queried
    violations, pages_processed = process_claim_pages(
        claim_path, pages, threads=THREADS, extended_coverages=extended_coverages
    )

    # Summarize the information for the claim
    summary_text = (
        summarize_results(violations) if len(violations) > 0 else "No violations found."
    )

    claim_summary = ClaimSummary(
        filepath=claim_path,
        pages_total=len(pages),
        pages_processed=len(pages_processed),
        pages_flagged=len(violations),
        summary=summary_text,
    )

    logging.info(f"Summary for {claim_path}:\n{summary_text}")

    return violations, claim_summary


@log_timer
def process_claims(
    is_cloud_run: bool,
    config_data_parameters: dict,
    run_id: str,
    claim_paths: list[str] = [],
    extended_coverage_dict: dict[str, list[ExtendedCoverage]] = {},
) -> None:
    """Processes a list of claims and outputs their violations and summaries to csv files.

    Main entrypoint for processing claims. First we apply the PageProcessor to identify
    potential violations in each claim then all violations are summarized. We save the
    violations and the summaries as .csv files.

    Args:
        is_cloud_run: True in case the execution started from the Google Colab Notebook, otherwise False.
        config_data_parameters: A dictionary containing OPENAI_API_KEY, Claims, Outputs and Logs folder paths.
        run_id: id to be appended to the beginning of all outputs such as logs and csv's.
        claim_paths: the paths of the files to be processed; if none are provided then
            all .pdf files in the CLAIMS_DIR will be processed.
        extended_coverage_dict: mapping from claims_path to extended coverages that were purchased
    """

    __configure_file_paths(is_cloud_run, config_data_parameters)

    log_path = os.path.join(LOGS_DIR, run_id + ".log")
    setup_logging(log_path=log_path)
    logging.info(f"Starting run {run_id}...")

    # Get list of all claims in claims directory if paths are not explicitly provided
    if not claim_paths:
        claim_paths = [
            os.path.join(CLAIMS_DIR, file)
            for file in os.listdir(CLAIMS_DIR)
            if file.endswith(".pdf")
        ]
    logging.info(f"All claims to be processed: {claim_paths}.")

    all_violations: list[Violation] = []
    all_summaries: list[ClaimSummary] = []

    for claim_path in claim_paths:
        extended_coverages = extended_coverage_dict.get(claim_path, [])
        violations, summary = process_single_claim(claim_path, extended_coverages)
        all_violations.extend(violations)
        all_summaries.append(summary)
        logging.info("---------------------------------------------\n")

    # Save the results
    output_base = os.path.join(OUTPUTS_DIR, run_id)
    pd.DataFrame(all_violations).to_csv(output_base + "_violations.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(output_base + "_summary.csv", index=False)
    logging.info("Done.")

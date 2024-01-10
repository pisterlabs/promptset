import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from claims_analysis.constants import (
    EXCLUDED_ITEMS_TEMPLATE,
    EXCLUDED_ITEMS_VIOLATION_TYPES,
    GLOBAL_EXCLUDED_KEYWORDS,
    PAIR_CLAUSE_TEMPLATE,
    PAIR_CLAUSE_VIOLATION_TYPES,
    RCV_PROPERTY_TEMPLATE,
    RCV_PROPERTY_VIOLATION_TYPES,
    YES_DELIMITER,
    ExtendedCoverage,
    ViolationType,
)
from claims_analysis.utils import words_exist_in_text


@dataclass
class Violation:
    """For storing individual occurences of violations."""

    filepath: str
    page_no: int
    issue_desc: str


class PageProcessor:
    """General processor for catching violations within a single page.

    Attributes:
        chat: Interface to ChatGPT client for sending queries to API
        required_keywords: A list of keywords to use for pre-filtering to determine if a
            page should be processed or not.
        sys_message: The initial message to prepend to all requests to ChatGPT, i.e. the system prompt.
    """

    def __init__(
        self,
        sys_message_template: str,
        relevant_violation_types: list[ViolationType],
        temperature: float = 0,
    ):
        """Initializes the instance based on the list of relevant violation types.

        Args:
            sys_message_template: prompt template to use for the initial system message
            relevant_violation_types: List of ViolationType containing description and relevant key words.
            temperature: Parameter between 0 and 1 controlling the randomness / creativity of the output.
                Closer to 0 makes the response more deterministic.
        """
        self.chat = ChatOpenAI(
            temperature=temperature, model_name="gpt-3.5-turbo", client=None
        )

        # Construct the base system message from the relevant violation types and save the keywords
        keywords_set: set[str] = set()
        violation_prompts: list[str] = []

        for violation_type in relevant_violation_types:
            violation_prompts.append(violation_type.prompt_desc)
            keywords_set.update(violation_type.keywords)
        self.required_keywords = list(keywords_set)

        violation_descriptions = "".join(
            "- " + desc + "\n" for desc in violation_prompts
        )
        self.sys_message = SystemMessage(
            content=sys_message_template.format(
                violation_descriptions=violation_descriptions,
                yes_delimiter=YES_DELIMITER,
            )
        )

    def meets_prefilter_criteria(self, page_text: str) -> bool:
        """Basic filter for ruling out pages that don't need to be processed."""

        # Terms that must be present to consider page
        if not words_exist_in_text(self.required_keywords, page_text):
            return False

        # Terms that must not be present to consider page; mostly terms in extended coverage doc
        return not words_exist_in_text(GLOBAL_EXCLUDED_KEYWORDS, page_text)

    def _process_response(self, raw_response: BaseMessage) -> Optional[str]:
        """Processes the response from LLM and returns a reason if there is one."""

        if raw_response.content.startswith(YES_DELIMITER):
            reason = raw_response.content.split(YES_DELIMITER)[-1].strip()
            return reason

        return None

    def process_page(self, page_text: str) -> Optional[str]:
        """Takes in a page and runs the LLM and returns a violation reason if there is one."""

        messages = [self.sys_message, HumanMessage(content=page_text)]
        response = self.chat(messages)

        return self._process_response(response)


def _filter_violation_types(
    violation_types: list[ViolationType], extended_coverages: list[ExtendedCoverage]
) -> list[ViolationType]:
    """Remove the violation types that the extended coverages cover."""
    return [
        violation_type
        for violation_type in violation_types
        if violation_type.extended_coverage not in extended_coverages
    ]


def process_claim_pages(
    path: str,
    pages: list[str],
    extended_coverages: list[ExtendedCoverage] = [],
    threads: int = 2,
) -> tuple[list[Violation], list[int]]:
    """Processes pages and returns a list of violations and page numbers that were processed.

    Args:
        path: path to the claim PDF file
        pages: the list of text of the claim pages
        extended_coverages: list of extended coverages that the policyholder has bought
        threads: number of concurrent workers for processing pages by Processors

    Returns:
        a list of potential violations and the total number of pages processed
    """

    logging.info(f"Starting processing for claim {path} with {threads} threads...")

    processors: list[PageProcessor] = []

    for prompt_template, viol_types in [
        # For excluded items (pool, patio)
        (EXCLUDED_ITEMS_TEMPLATE, EXCLUDED_ITEMS_VIOLATION_TYPES),
        # For RCV with non-covered properties
        (RCV_PROPERTY_TEMPLATE, RCV_PROPERTY_VIOLATION_TYPES),
        # For items covered under the pair and set clause
        (PAIR_CLAUSE_TEMPLATE, PAIR_CLAUSE_VIOLATION_TYPES),
    ]:
        if filt_types := _filter_violation_types(viol_types, extended_coverages):
            processors.append(
                PageProcessor(
                    sys_message_template=prompt_template,
                    relevant_violation_types=filt_types,
                )
            )

    pages_processed: set[int] = set()
    violations: list[Violation] = []

    # This is a list of tuples where the first element is the page number and the second element is a future of the
    # chat gpt result.
    page_processing_future: list[tuple[int, Future[Optional[str]]]] = []

    # Submit the pages to processor. Although all threads land on 1 CPU in Pyhton, this will offer a speedup
    # since we're bottlenecked by each network request and processing by ChatGPT, not our own internal processing.
    with ThreadPoolExecutor(max_workers=threads) as exec:
        for page_no, page in enumerate(pages, 1):
            for processor in processors:
                if processor.meets_prefilter_criteria(page):
                    pages_processed.add(page_no)
                    page_processing_future.append(
                        (page_no, exec.submit(processor.process_page, page))
                    )

    # Collect the results
    for page_no, future in page_processing_future:
        if reason := future.result():
            violations.append(
                Violation(filepath=path, page_no=page_no, issue_desc=reason)
            )
            logging.info(f"Found violation on page {page_no} with reason: {reason}")

    logging.info(
        f"Finished {path}. Processed {len(pages_processed)} pages out of {len(pages)}: {pages_processed}"
    )

    return violations, list(pages_processed)

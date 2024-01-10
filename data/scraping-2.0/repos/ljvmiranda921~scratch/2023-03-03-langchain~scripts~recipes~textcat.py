"""Module that contains Prodigy Recipes for LLM-assisted Textcat annotation"""

import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import spacy
import srsly
from dotenv import load_dotenv
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.docstore.document import Document
from langchain.document_loaders import PagedPDFSplitter
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.text_splitter import SpacyTextSplitter
from prodigy.components import preprocess
from prodigy.components.loaders import get_stream
from prodigy.core import recipe
from prodigy.types import TaskType
from prodigy.util import log, msg, set_hashes
from spacy.language import Language
from tqdm import tqdm

from scripts.recipes.langchain import load_prodigy_chain
from scripts.parsers import get_parser, LABELS


@recipe(
    # fmt: off
    "textcat.langchain.fetch",
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    output_path=("Path to save the output", "positional", None, Path),
    annotation_guideline=("Path to the PDF annotation guideline", "option", "G", Path),
    lang=("Language to initialize spaCy model", "option", "l", str),
    chain_type=("Prompt-style to use for combining documents. Choose from 'stuff', 'map_reduce', 'map_rerank', or 'refine'", "option", "C", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    segment=("Split sentences passed to the source", "flag", "S", bool),
    temperature=("Temperature parameter to control LLM generation", "option", "t", float),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    resume=("Resume fetch from the output file", "flag", "r", bool),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    # fmt: on
)
def langchain_textcat_fetch(
    source: Union[str, Iterable[Dict]],
    output_path: Path,
    annotation_guideline: Path,
    lang: str = "en",
    chain_type: str = "stuff",
    model: str = "text-davinci-003",
    segment: bool = False,
    temperature: float = 0.7,
    batch_size: int = 10,
    resume: bool = False,
    loader: Optional[str] = None,
):
    """Perform bulk zero-shot annotation using GPT-3 with the aid of an
    annotation guideline. For binary text classification only.

    Because annotation guidelines tend to be long, the document is split into
    chunks and fed piecemeal to an LLM's prompt. You can control how these
    chunks (and their respective outputs) are combined together by setting the
    `--chain-type` parameter. It has four options: `stuff`, `map_reduce`,
    `map_rerank`, and `refine.`

    You can find more information in the Langchain documentation:
    https://langchain.readthedocs.io/en/latest/modules/indexes/combine_docs.html
    """
    log("RECIPE: Starting recipe langchain.textcat.fetch", locals())
    nlp = spacy.blank(lang)
    if segment:
        nlp.add_pipe("sentencizer")

    # Setup and validate the annotation guideline's filepath
    if not annotation_guideline.exists():
        msg.fail(
            f"Cannot find path to the annotation guideline ({annotation_guideline})",
            exits=0,
        )
    msg.text(f"Using annotation guideline from {annotation_guideline}")
    pages = load_document(annotation_guideline)

    # Setup OpenAI
    api_key, _ = get_api_credentials()
    llm = OpenAI(openai_api_key=api_key, model_name=model, temperature=temperature)

    # Setup the stream and the Prodigy UI
    suggester = Suggester(
        llm,
        pages,
        segment=segment,
        response_parser=get_parser(annotation_guideline),
    )
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )

    # If we want to resume, we take the path to the cache and
    # compare the hashes with respect to our inputs.
    if resume:
        msg.info(f"Resuming from previous output file: {output_path}")
        stream = get_resume_stream(stream, srsly.read_jsonl(output_path))

    # Run the LangChain suggester on the stream
    stream = suggester(
        tqdm(stream),
        nlp=nlp,
        batch_size=batch_size,
        chain_type=chain_type,
    )
    srsly.write_jsonl(output_path, stream, append=resume, append_new_line=False)


@recipe(
    # fmt: off
    "textcat.langchain.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    annotation_guideline=("Path to the PDF annotation guideline", "option", "G", Path),
    lang=("Language to initialize spaCy model", "option", "l", str),
    chain_type=("Prompt-style to use for combining documents. Choose from 'stuff', 'map_reduce', 'map_rerank', or 'refine'", "option", "C", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    segment=("Split sentences passed to the source", "flag", "S", bool),
    temperature=("Temperature parameter to control LLM generation", "option", "t", float),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    # fmt: on
)
def langchain_textcat_correct(
    dataset: str,
    source: Union[str, Iterable[Dict]],
    annotation_guideline: Path,
    lang: str = "en",
    chain_type: str = "stuff",
    model: str = "text-davinci-003",
    segment: bool = False,
    temperature: float = 0.7,
    batch_size: int = 10,
    loader: Optional[str] = None,
):
    """Perform bulk zero-shot annotation using GPT-3 with the aid of an
    annotation guideline. For binary text classification only.

    Because annotation guidelines tend to be long, the document is split into
    chunks and fed piecemeal to an LLM's prompt. You can control how these
    chunks (and their respective outputs) are combined together by setting the
    `--chain-type` parameter. It has four options: `stuff`, `map_reduce`,
    `map_rerank`, and `refine.`

    You can find more information in the Langchain documentation:
    https://langchain.readthedocs.io/en/latest/modules/indexes/combine_docs.html
    """
    log("RECIPE: Starting recipe langchain.textcat.fetch", locals())
    nlp = spacy.blank(lang)
    if segment:
        nlp.add_pipe("sentencizer")

    # Setup and validate the annotation guideline's filepath
    if not annotation_guideline.exists():
        msg.fail(
            f"Cannot find path to the annotation guideline ({annotation_guideline})",
            exits=0,
        )
    msg.text(f"Using annotation guideline from {annotation_guideline}")
    pages = load_document(annotation_guideline)

    # Setup OpenAI
    api_key, _ = get_api_credentials()
    llm = OpenAI(openai_api_key=api_key, model_name=model, temperature=temperature)

    # Setup the stream and the Prodigy UI
    suggester = Suggester(
        llm, pages, segment=segment, response_parser=get_parser(annotation_guideline)
    )
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )

    # Run the LangChain suggester on the stream
    stream = suggester(
        tqdm(stream),
        nlp=nlp,
        batch_size=batch_size,
        chain_type=chain_type,
        get_rank_ctx=True,
    )

    # Set up the Prodigy UI
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "config": {
            "labels": LABELS.get(annotation_guideline.stem).get("labels"),
            "batch_size": batch_size,
            "exclude_by": "input",
            "choice_style": "single",
            "blocks": [
                {"view_id": "choice"},
                # {"view_id": "html", "html_template": HTML_TEMPLATE},  TODO
            ],
            # "global_css": GLOBAL_STYLE,  TODO
        },
    }


def get_api_credentials() -> Tuple[str, str]:
    """Obtain OpenAI API credentials from a .env file"""
    load_dotenv()
    api_key = os.getenv("PRODIGY_OPENAI_KEY")
    api_org = os.getenv("PRODIGY_OPENAI_ORG")
    if api_key is None or api_org is None:
        msg.fail("Can't find API credentials from the environment.", exit=1)
    return api_key, api_org


def load_document(file_path: Path) -> List[Document]:
    """Load PDF document and return its pages"""
    loader = PagedPDFSplitter(str(file_path))
    pages = loader.load_and_split(
        text_splitter=SpacyTextSplitter(pipeline="en_core_web_sm")
    )
    return pages


def get_resume_stream(stream: Iterable[Dict], cache: Iterable[TaskType]):
    # Get all hashes in the cache
    cache_ids = [eg.get("_input_hash") for eg in cache]
    log(f"Found {len(cache_ids)} hashes in cache")
    # Hash the current stream and return examples not in cache
    hashed_stream = [(set_hashes(eg).get("_input_hash"), eg) for eg in stream]
    resume = [eg for _hash, eg in hashed_stream if _hash not in cache_ids]
    return resume


class Suggester:
    """Suggester that takes an LLM class and the processed document to suggest labels"""

    def __init__(
        self,
        llm: BaseLLM,
        pages: List[Document],
        *,
        response_parser: Callable,
        segment: bool,
    ):
        self.llm = llm
        self.pages = pages
        self.segment = segment
        self.response_parser = response_parser

    def __call__(
        self,
        stream: Iterable[Dict],
        *,
        nlp: Language,
        batch_size: int,
        chain_type: str,
        get_rank_ctx: bool = False,
    ) -> Iterable[TaskType]:
        if self.segment:
            stream = preprocess.split_sentences(nlp, stream)

        chain = load_prodigy_chain(llm=self.llm, chain_type=chain_type)

        # Create a chain for getting a ranked list
        rank_chain = (
            load_prodigy_chain(llm=self.llm, chain_type="map_rerank")
            if get_rank_ctx
            else None
        )

        stream = self.pipe(stream, nlp, batch_size, chain, rank_chain)
        stream = self.set_hashes(stream)
        return stream

    def pipe(
        self,
        stream: Iterable[TaskType],
        nlp: Language,
        batch_size: int,
        chain: BaseCombineDocumentsChain,
        rank_chain: Optional[BaseCombineDocumentsChain],
    ) -> Iterable[TaskType]:
        """Process the stream and add suggestions"""
        stream = self.stream_suggestions(stream, batch_size, chain, rank_chain)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def stream_suggestions(
        self,
        stream: Iterable[TaskType],
        batch_size: int,
        chain: BaseCombineDocumentsChain,
        rank_chain: Optional[BaseCombineDocumentsChain],
    ) -> Iterable[TaskType]:
        """Split the stream into batches and get the response from LangChain"""
        for batch in self._batch_sequence(stream, batch_size):
            responses: List[str] = [
                chain.run(question=eg.get("text", ""), context=self.pages)
                for eg in batch
            ]

            if rank_chain:
                relevant_ctx = [
                    rank_chain.combine_docs(
                        question=eg.get("text", ""), docs=self.pages
                    )
                    for eg in batch
                ]
                breakpoint()
            else:
                relevant_ctx = []

            for eg, response in zip(batch, responses):
                eg["llm"] = {"response": response, "relevant_ctx": relevant_ctx}
                yield eg

    def set_hashes(self, stream: Iterable[TaskType]) -> Iterable[TaskType]:
        for example in stream:
            yield set_hashes(example)

    def format_suggestions(
        self, stream: Iterable[TaskType], nlp: Language
    ) -> Iterable[TaskType]:
        """Parse the output from the large language model"""
        stream = preprocess.add_tokens(nlp, stream, skip=True)
        for example in stream:
            example = copy.deepcopy(example)
            if "meta" not in example:
                example["meta"] = {}

            response = example["llm"].get("response", "")
            example.update(self.response_parser(response))
            yield example

    def _batch_sequence(
        self, items: Iterable[TaskType], batch_size: int
    ) -> Iterable[List[TaskType]]:
        batch = []
        for eg in items:
            batch.append(eg)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

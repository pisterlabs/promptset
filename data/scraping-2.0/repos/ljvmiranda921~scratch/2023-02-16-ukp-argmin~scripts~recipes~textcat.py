from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import spacy
import srsly
from prodigy.components.loaders import get_stream
from prodigy.core import recipe
from prodigy.util import log, msg
from tqdm import tqdm

from scripts.recipes.openai import GLOBAL_STYLE, OPENAI_DEFAULTS
from scripts.recipes.openai import OpenAISuggester, PromptExample, _ItemT
from scripts.recipes.openai import get_api_credentials, get_resume_stream
from scripts.recipes.openai import load_template, normalize_label
from scripts.recipes.openai import read_prompt_examples


@dataclass
class TextCatPromptExample(PromptExample):
    """An example to be passed into an OpenAI TextCat prompt."""

    text: str
    answer: str
    reason: str

    @classmethod
    def from_prodigy(cls, example: _ItemT, labels: List[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format."""
        if "text" not in example:
            raise ValueError("Cannot make PromptExample without text")

        full_text = example["text"]
        reason = example["meta"].get("reason")
        if len(labels) == 1:
            answer = example.get("answer", "reject")
        else:
            answer = ",".join(example.get("accept", []))
        return cls(text=full_text, answer=answer, reason=reason)


def make_textcat_response_parser(labels: List[str]) -> Callable:
    def _parse_response(text: str, example: Optional[Dict] = None) -> Dict:
        response: Dict[str, str] = {}
        if text and any(k in text.lower() for k in ("answer", "reason")):
            for line in text.strip().split("\n"):
                if line and ":" in line:
                    key, value = line.split(":", 1)
                    # To make parsing easier, we normalize the answer returned by
                    # OpenAI (lower-case). We also normalize the labels so that we
                    # can match easily.
                    response[key.strip().lower()] = normalize_label(value.strip())
        else:
            response = {"answer": "", "reason": ""}

        example = _fmt_binary(response) if len(labels) == 1 else _fmt_multi(response)
        return example

    def _fmt_binary(response: Dict[str, str]) -> Dict:
        """Parse binary TextCat where the 'answer' key means it's a positive class."""
        return {
            "answer": response["answer"].lower(),
            "label": labels[0],
            "meta": {
                "answer": response["answer"].upper(),
                "reason": response["reason"],
            },
        }

    def _fmt_multi(response: Dict[str, str]) -> Dict:
        """Parse multilabel TextCat where the 'accept' key is a list of positive labels."""
        return {
            "options": [{"id": label, "text": label} for label in labels],
            "answer": "accept",
            "meta": {
                "reason": response.get("reason", ""),
                "GPT-3_answer": response.get("answer", ""),
            },
            "accept": list(
                filter(None, [s.strip() for s in response["answer"].split(",")])
            ),
        }

    return _parse_response


@recipe(
    # fmt: off
    "textcat.openai.correct",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to initialize spaCy model", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    prompt_path=("Path to the .jinja2 prompt template", "option", "p", str),
    examples_path=("Examples file to help define the task", "option", "e", str),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def openai_correct_textcat(
    dataset: str,
    source: Union[str, Iterable[dict]],
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    prompt_path: Path = OPENAI_DEFAULTS.TEXTCAT_PROMPT_PATH,
    examples_path: Optional[Path] = None,
    max_examples: int = 2,
    exclusive_classes: bool = False,
    loader: Optional[str] = None,
    verbose: bool = False,
):
    """
    Perform zero- or few-shot annotation with the aid of GPT-3. Prodigy will
    infer binary or multilabel classification based on the number of labels in
    --labels. You can also set the -E flag to make the classification task exclusive.
    """
    log("RECIPE: Starting recipe textcat.openai.correct", locals())
    api_key, api_org = get_api_credentials(model)
    examples = read_prompt_examples(examples_path, example_class=TextCatPromptExample)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    if not labels:
        msg.fail("No --label argument set", exits=1)
    msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")

    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    # Create OpenAISuggester with GPT-3 parameters
    labels = [normalize_label(label) for label in labels]
    openai = OpenAISuggester(
        response_parser=make_textcat_response_parser(labels=labels),
        prompt_template=load_template(prompt_path),
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_model=model,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        render_vars={"exclusive_classes": exclusive_classes},
        prompt_example_class=TextCatPromptExample,
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)

    # Set up the stream
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )
    stream = openai(stream, batch_size=batch_size, nlp=nlp)

    # Set up the Prodigy UI
    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "update": openai.update,
        "config": {
            "labels": openai.labels,
            "batch_size": batch_size,
            "exclude_by": "input",
            "choice_style": "single" if exclusive_classes else "multiple",
            "blocks": [
                {"view_id": "classification" if len(labels) == 1 else "choice"},
                {
                    "view_id": "html",
                    "html_template": OPENAI_DEFAULTS.TEXTCAT_HTML_TEMPLATE,
                },
            ],
            "show_flag": True,
            "global_css": GLOBAL_STYLE,
        },
    }


@recipe(
    # fmt: off
    "textcat.openai.fetch",
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    output_path=("Path to save the output", "positional", None, Path),
    labels=("Labels (comma delimited)", "option", "L", lambda s: s.split(",")),
    lang=("Language to use for tokenizer.", "option", "l", str),
    model=("GPT-3 model to use for completion", "option", "m", str),
    prompt_path=("Path to jinja2 prompt template", "option", "p", str),
    examples_path=("Examples file to help define the task", "option", "e", str),
    max_examples=("Max examples to include in prompt", "option", "n", int),
    batch_size=("Batch size to send to OpenAI API", "option", "b", int),
    segment=("Split sentences", "flag", "S", bool),
    exclusive_classes=("Make the classification task exclusive", "flag", "E", bool),
    resume=("Resume fetch by passing a path to a cache", "flag", "r", bool),
    verbose=("Print extra information to terminal", "flag", "v", bool),
    # fmt: on
)
def openai_fetch_textcat(
    source: str,
    output_path: Path,
    labels: List[str],
    lang: str = "en",
    model: str = "text-davinci-003",
    batch_size: int = 10,
    segment: bool = False,
    prompt_path: Path = OPENAI_DEFAULTS.TEXTCAT_PROMPT_PATH,
    examples_path: Optional[Path] = None,
    max_examples: int = 2,
    exclusive_classes: bool = False,
    resume: bool = False,
    loader: Optional[str] = None,
    verbose: bool = False,
):
    """
    Get bulk textcat suggestions from the OpenAI API, using zero-shot or
    few-shot learning. The results can then be corrected using the
    `textcat.manual` recipe.  This approach lets you get OpenAI queries upfront,
    which can help if you want multiple annotators or reduce the waiting time
    between API calls.
    """
    log("RECIPE: Starting recipe textcat.openai.fetch", locals())
    api_key, api_org = get_api_credentials(model)
    examples = read_prompt_examples(examples_path, example_class=TextCatPromptExample)
    nlp = spacy.blank(lang)

    if segment:
        nlp.add_pipe("sentencizer")

    if not labels:
        msg.fail("No --label argument set", exits=1)
    msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")

    if not exclusive_classes and len(labels) == 1:
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    # Create OpenAISuggester with GPT-3 parameters
    labels = [normalize_label(label) for label in labels]
    openai = OpenAISuggester(
        response_parser=make_textcat_response_parser(labels=labels),
        prompt_template=load_template(prompt_path),
        labels=labels,
        max_examples=max_examples,
        segment=segment,
        openai_api_org=api_org,
        openai_api_key=api_key,
        openai_n=1,
        openai_model=model,
        openai_retry_timeout_s=10,
        openai_read_timeout_s=20,
        openai_n_retries=10,
        render_vars={"exclusive_classes": exclusive_classes},
        prompt_example_class=TextCatPromptExample,
        verbose=verbose,
    )
    for eg in examples:
        openai.add_example(eg)

    # Set up the stream
    stream = get_stream(
        source, loader=loader, rehash=False, dedup=False, input_key="text"
    )
    # If we want to resume, we take the path to the cache and
    # compare the hashes with respect to our inputs.
    if resume:
        msg.info(f"Resuming from previous output file: {output_path}")
        stream = get_resume_stream(stream, srsly.read_jsonl(output_path))

    stream = openai(tqdm(stream), batch_size=batch_size, nlp=nlp)
    srsly.write_jsonl(output_path, stream, append=resume, append_new_line=False)

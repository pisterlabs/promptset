import json
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Generator, Optional
from joblib import Memory

import openai
import rich
from pydantic import BaseModel, ValidationError
from retry import retry

from ..utils import (
    chunks,
    flatten,
    get_jinja_environment,
    get_project_cache,
    get_project_root,
    load_jsonl,
)
from .eval_results import clean_string_units
from .search import (
    PageSearchOutput,
    get_non_overlapping_chunks,
    nearest_pages,
    page_coverage,
)

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

extraction_chat_completion_tmpl = get_jinja_environment().get_template(
    "extraction_chat_completion.pmpt.tpl"
)
extraction_completion_tmpl = get_jinja_environment().get_template(
    "extraction_completion.pmpt.tpl"
)

memory = Memory(get_project_root() / ".joblib_cache", verbose=2)


class District(BaseModel):
    T: str
    Z: str


class PromptOutput(BaseModel):
    answer: str
    extracted_text: list[str]
    pages: list[int]
    confidence: float


class LookupOutput(BaseModel):
    output: Optional[PromptOutput]
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


class AllLookupOutput(BaseModel):
    town: str
    district: District
    sizes: dict[str, list[LookupOutput]]


TEMPLATE_MAPPING = {
    "text-davinci-003": extraction_completion_tmpl,
    "gpt-3.5-turbo": extraction_chat_completion_tmpl,
    "gpt-4": extraction_chat_completion_tmpl,
}


@memory.cache
@retry(exceptions=(openai.error.APIError, openai.error.RateLimitError), tries=-1, delay=10, backoff=1.25, jitter=(1, 10))  # type: ignore
def lookup_term_prompt(
    model_name: str, page_text, district, term
) -> PromptOutput | None:
    match model_name:
        case "text-davinci-003":
            resp = openai.Completion.create(
                model=model_name,
                max_tokens=256,
                prompt=TEMPLATE_MAPPING[model_name].render(
                    passage=page_text,
                    term=term,
                    synonyms=", ".join(thesaurus.get(term, [])),
                    zone_name=district["T"],
                    zone_abbreviation=district["Z"],
                ),
            )
            top_choice = resp.choices[0]
            text = top_choice.text
        case "gpt-3.5-turbo" | "gpt-4":
            resp = openai.ChatCompletion.create(
                model=model_name,
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": TEMPLATE_MAPPING[model_name].render(
                            term=term,
                            synonyms=", ".join(thesaurus.get(term, [])),
                            zone_name=district["T"],
                            zone_abbreviation=district["Z"],
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Input: \n\n {page_text}\n\n Output:",
                    },
                ],
            )
            top_choice = resp.choices[0]
            text = top_choice.message.content
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    try:
        json_body = json.loads(text)
        if json_body is None:
            # The model is allowed to return null if it cannot find the answer,
            # so just pass this onwards.
            return None
        return PromptOutput(**json_body)
    except (ValidationError, TypeError, json.JSONDecodeError) as exc:
        rich.print("Error parsing response from model during extraction:", exc)
        rich.print(f"Response: {text}")
        return None


class ExtractionMethod(str, Enum):
    NONE = "search_only"
    STUFF = "stuff"
    MAP = "map"


def extract_size(
    town: str,
    district: District,
    term: str,
    top_k_pages: int,
    method: ExtractionMethod = ExtractionMethod.MAP,
    model_name: str = "text-davinci-003",
) -> list[LookupOutput]:
    pages = nearest_pages(town, district, term)
    pages = get_non_overlapping_chunks(pages)[:top_k_pages]

    if len(pages) == 0:
        return []

    outputs = []
    match method:
        case ExtractionMethod.NONE:
            for page in pages:
                outputs.append(
                    LookupOutput(
                        output=None,
                        search_pages=[page],
                        search_pages_expanded=flatten(page_coverage([page])),
                    )
                )
        case ExtractionMethod.STUFF:
            # Stuff all pages into prompt, in order of page number
            all_page = reduce(
                lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
            )
            # This is the length of the prompt before any template interpolation
            # TODO: Determine this automatically
            prompt_base_token_length = 256
            for chunk in chunks(all_page, 8192 - prompt_base_token_length):
                outputs.append(
                    LookupOutput(
                        output=lookup_term_prompt(model_name, chunk, district, term),
                        search_pages=pages,
                        search_pages_expanded=flatten(page_coverage(pages)),
                    )
                )
        case ExtractionMethod.MAP:
            with ThreadPoolExecutor(max_workers=20) as executor:
                for page, result in executor.map(
                    lambda page: (
                        page,
                        lookup_term_prompt(model_name, page.text, district, term),
                    ),
                    pages,
                ):
                    outputs.append(
                        LookupOutput(
                            output=result,
                            search_pages=[page],
                            search_pages_expanded=flatten(page_coverage([page])),
                        )
                    )

    return sorted(
        outputs, key=lambda x: x.output.confidence if x.output else 0, reverse=True
    )


def extract_all_sizes(
    town_districts: list[dict], terms: list[str], top_k_pages: int
) -> Generator[AllLookupOutput, None, None]:
    for d in town_districts:
        town = d["Town"]
        districts = d["Districts"]

        for district in districts:
            yield AllLookupOutput(
                town=town,
                district=district,
                sizes={
                    term: extract_size(
                        town, district, term, top_k_pages, method=ExtractionMethod.MAP
                    )
                    for term in terms
                },
            )


def main():
    districts_file = get_project_root() / "data" / "results" / "districts_gt.jsonl"
    import pandas as pd

    gt = pd.read_csv(
        get_project_root() / "data" / "ground_truth.csv",
        index_col=["town", "district_abb"],
    )

    town_districts = load_jsonl(districts_file)

    for result in extract_all_sizes(
        town_districts, ["min lot size", "min unit size"], 6
    ):
        for term, lookups in result.sizes.items():
            for l in lookups:
                expected = set(
                    float(f)
                    for f in gt.loc[
                        result.town, result.district.Z
                    ].min_lot_size_gt.split(", ")
                )
                actual = (
                    set(clean_string_units(l.output.answer))
                    if l.output is not None
                    else set()
                )
                is_correct = any(expected & actual)
                if not is_correct:
                    print(
                        f"{result.town} - {result.district.T} ({l.output.pages}): {term} | Expected: {expected} | Actual: {actual} | Correct: {is_correct}"
                    )


if __name__ == "__main__":
    main()

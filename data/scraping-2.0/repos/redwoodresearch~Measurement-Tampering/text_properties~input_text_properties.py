import json
import re
import threading
import time
from typing import Any, Optional

import attrs
import openai
import tiktoken
from tqdm import tqdm

tokenizer = tiktoken.encoding_for_model("gpt-4")

example_properties = ["strictly more than 3 lines", "strictly more than 30 lines", "the word 'I'"]

question_prefix = "Does the text contain"


def get_questions_prompt(all_input_properties: list[str]):
    return "\n".join(f"{i+1}. {question_prefix} {prop}?" for i, prop in enumerate(all_input_properties)) + "\nAnswers:"


def get_remaining_questions(all_input_properties: list[str]):
    return "Now for the actual properties.\n\n" + get_questions_prompt(all_input_properties)


basic_prompt = """Here is some text:
```
{}
```

I'm interested in the properties of this text. In particular, answer Yes/No to the following questions in an enumerating list.
First, repeat the question, then provide the answer. The answer should just be Yes/No without *any* additional text.

I'll first ask about some example properties and then do the actual properties.

""" + get_questions_prompt(
    example_properties
)


@attrs.frozen
class QueryItem:
    cut_text: str
    query: Optional[list[dict[str, str]]]  # allow noop query
    full_text: str
    extra: dict[str, Any] = attrs.Factory(dict)

    @property
    def query_unwrap(self):
        q = self.query
        assert q is not None
        return q


def cut_text(x: str, target_tokens: int = 300):
    if "```" in x:
        # avoid backticks
        return None
    out_lines = []
    tok_count_total = 0
    for line in x.splitlines():
        line_count = len(tokenizer.encode(line))
        if line_count + tok_count_total > target_tokens:
            break

        out_lines.append(line)
        tok_count_total += line_count

    while out_lines and out_lines[-1] == "":
        out_lines = out_lines[:-1]

    if tok_count_total < target_tokens // 2 or len(out_lines) < 3:
        return None

    return "\n".join(out_lines)


QueryItems = list[QueryItem]


def make_query(new_text: str, props_to_check: list[str], full_text: str):
    if len(props_to_check) > 0:
        full_prompt = str(basic_prompt).format(new_text)
        out_lines = new_text.splitlines()
        props = [
            len(out_lines) > 3,
            len(out_lines) > 30,
            re.search(r"\bI\b", new_text) is not None,
        ]
        example_response = "\n".join(
            f"{i+1}. {question_prefix} {prop_text}? {'Yes' if prop else 'No'}"
            for i, (prop, prop_text) in enumerate(zip(props, example_properties))
        )

        actual_prompt = [
            {"role": "user", "content": full_prompt},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": get_remaining_questions(props_to_check)},
        ]
    else:
        actual_prompt = None

    return QueryItem(new_text, actual_prompt, full_text)


def data_to_queries(data: list, props_to_check: list[str], limit: int = 999999):
    query_items: QueryItems = []

    count = 0
    last_datum_use = 0
    for i, d in enumerate(data):
        if count == limit:
            break
        new_text = cut_text(d["text"])
        if new_text is None:
            continue
        count += 1

        query_items.append(make_query(new_text, props_to_check, d["text"]))

        last_datum_use = i + 1

    return query_items, last_datum_use


pbar_lock = threading.Lock()


def get_completions(
    all_queries: QueryItems,
    k: int,
    n_threads: int,
    models: list[str] = ["gpt-3.5-turbo-0613"],
    out_file: str = "temp.jsonl",
    max_tokens: int = 3000,
    extra_args: dict[str, Any] = {},
    pbar: Optional[tqdm] = None,
    max_retries: int = 10,
):
    start = k * len(all_queries) // n_threads
    end = (k + 1) * len(all_queries) // n_threads
    tmp_responses: list[dict[str, Any]] = []
    tmp_queries: QueryItems = []
    all_queries = all_queries[start:end]
    print(f"{k=} {n_threads=} {start=} {end=}")

    def dump_and_clear(item: str):
        with open(out_file, "a") as f:
            for query, response in zip(tmp_queries, tmp_responses):
                f.write(
                    json.dumps(
                        {
                            "query": attrs.asdict(query),
                            "response": response,
                        }
                    )
                    + "\n"
                )
        print(f"thread {k} {item} done")
        tmp_responses.clear()
        tmp_queries.clear()

    for i in range(len(all_queries)):
        query = all_queries[i]
        responses_per_m = {}
        for model in models:
            retries = 0
            while True:
                try:
                    if query.query is None:
                        responses_per_m[model] = None
                        break
                    time.sleep(min(0.2 * 2**retries, 10.0))
                    responses_per_m[model] = openai.ChatCompletion.create(
                        model=model,
                        messages=query.query,
                        temperature=0,
                        max_tokens=max_tokens,
                        **extra_args,
                    )
                    break
                except Exception as e:
                    print("error")
                    print(e)
                    print(f"{retries=}")
                    if retries > max_retries:
                        print("exceeded max retries, exiting")
                        break
                    retries += 1

        if len(responses_per_m) == 0:
            continue

        tmp_responses.append(responses_per_m)
        tmp_queries.append(all_queries[i])

        # if i % 10 == 0 and i > 0:
        dump_and_clear(str(i))

        if k == 0 and query.query is not None:
            print()
            print(responses_per_m[models[0]]["choices"][0]["message"]["content"])
            print(responses_per_m[models[0]]["choices"][0]["message"].get("function_call"))

        if pbar is not None:
            with pbar_lock:
                if pbar is not None:
                    pbar.update(1)

    dump_and_clear("fin")

import json
import threading
import time
from typing import Any, Optional

import attrs
import openai
from tqdm import tqdm

basic_writing_prompt = """Here is some text:
```
{}
```

Modify this text so that it contains {}. Just output the new text without any commentary."""

pbar_lock = threading.Lock()


def remove_backticks_and_strip(x: str):
    if x.count("```") == 1:
        backtick_loc = x.index("```")
        if backtick_loc < len(x) / 3:
            # assume leading
            return x[backtick_loc + 3 :].strip("\n")
        if backtick_loc > 2 * len(x) / 3:
            # assume trailing
            return x[:backtick_loc].strip("\n")

    if x.count("```") == 2:
        x = x[x.index("```") + 3 :]
        x = x[: x.index("```")]
        x = x.strip("\n")
        return x

    return x.strip("\n")


@attrs.frozen
class RewriteItem:
    cut_text: str
    modifications: list[str]
    full_text: str
    extra: dict[str, Any] = attrs.Factory(dict)


def run_modifications(
    model: str, temperature: float, max_tokens: int, rewrite: RewriteItem, printing: bool, max_retries: int = 10
):
    last_text = rewrite.cut_text
    all_text = [last_text]
    for i, mod in enumerate(rewrite.modifications):
        this_w_prompt = str(basic_writing_prompt).format(last_text, mod)

        prior_mod = rewrite.modifications[:i]

        if len(prior_mod) != 0:
            this_w_prompt += "\nWhen modifying the text, retain the following properties of the text:\n" + "\n".join(
                f"- {p_mod}" for p_mod in prior_mod
            )

        this_w_prompt += "\n\nModified text:"

        retries = 0
        while True:
            try:
                time.sleep(min(0.2 * 2**retries, 10.0))

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": this_w_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
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

        new_text = remove_backticks_and_strip(response["choices"][0]["message"]["content"])
        all_text.append(new_text)
        last_text = new_text

        if printing:
            print("\n" * 10)
            print(this_w_prompt)
            print("=== resp ===")
            print(new_text)

    return all_text, last_text


def get_rewrites(
    all_rewrites: list[RewriteItem],
    k: int,
    n_threads: int,
    models: list[str] = ["gpt-3.5-turbo-0613"],
    out_file: str = "temp.jsonl",
    max_tokens: int = 1_000,
    temperature=0.3,
    pbar: Optional[tqdm] = None,
):
    start = k * len(all_rewrites) // n_threads
    end = (k + 1) * len(all_rewrites) // n_threads
    tmp_all_text: list[list[str]] = []
    tmp_final_text: list[str] = []
    tmp_model: list[str] = []
    tmp_queries: list[RewriteItem] = []
    all_rewrites = all_rewrites[start:end]
    print(f"{k=} {n_threads=} {start=} {end=}")

    def dump_and_clear(item: str):
        with open(out_file, "a") as f:
            for query, all_text, final_text, model in zip(tmp_queries, tmp_all_text, tmp_final_text, tmp_model):
                f.write(
                    json.dumps(
                        {
                            "query": attrs.asdict(query),
                            "all_text": all_text,
                            "final_text": final_text,
                            "model": model,
                        }
                    )
                    + "\n"
                )
        print(f"thread {k} {item} done")
        tmp_all_text.clear()
        tmp_final_text.clear()
        tmp_model.clear()
        tmp_queries.clear()

    for i in range(len(all_rewrites)):
        for model in models:
            out_mod = run_modifications(
                model,
                temperature=temperature,
                max_tokens=max_tokens,
                rewrite=all_rewrites[i],
                printing=k == 0,
            )
            if out_mod is None:
                continue
            all_text, final_text = out_mod

            # pass
            tmp_all_text.append(all_text)
            tmp_final_text.append(final_text)
            tmp_model.append(model)
            tmp_queries.append(all_rewrites[i])

            # if i % 10 == 0 and i > 0:
            dump_and_clear(str(i))

            if k == 0:
                print()
                print(model)
                print(final_text)

            if pbar is not None:
                with pbar_lock:
                    if pbar is not None:
                        pbar.update(1)

    dump_and_clear("fin")

import csv
from typing import Any

import backoff
import openai

FREQ_LIST_PATH = "../../smart_word_hints_api/app/assets/amalgum_freq_list.csv"
NUMBER_OF_MOST_COMMON_DEFINITIONS_TO_SIMPLIFY = 22857
OUTPUT_PATH = "out.csv"
OUTPUT_HEADER = "key|lemma|pos|simple_definition|original_definition\n"
GPT_PROMPT_TEMPLATE = (
    "long definition of {lemma}: {wordnet_definition}\n"
    "extra short definition for an English learner: "
)


def load_freq_list() -> list[dict[str, Any]]:
    with open(FREQ_LIST_PATH, newline="", encoding="utf_8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        return list(reader)


def definitions_already_simplified() -> set[str]:
    with open(OUTPUT_PATH, newline="", encoding="utf_8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        return set([row["key"] for row in reader])


def create_output_if_doesnt_exist() -> None:
    try:
        with open(OUTPUT_PATH, "r", encoding="utf_8") as f:
            if f.readline() != OUTPUT_HEADER:
                raise FileNotFoundError(f"Output file {OUTPUT_PATH} incorrect")
    except FileNotFoundError:
        with open(OUTPUT_PATH, "w", encoding="utf_8") as f:
            f.write(OUTPUT_HEADER)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completion_with_backoff(gpt_prompt: str):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=gpt_prompt,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


def main():
    freq_list = load_freq_list()
    create_output_if_doesnt_exist()
    already_simplified = definitions_already_simplified()
    with open(OUTPUT_PATH, "a", encoding="utf_8") as f:
        output_writer = csv.writer(f, delimiter="|")
        for row in freq_list[:NUMBER_OF_MOST_COMMON_DEFINITIONS_TO_SIMPLIFY]:
            if row["key"] in already_simplified:
                continue
            gpt_prompt = GPT_PROMPT_TEMPLATE.format(
                lemma=row["lemma"].replace("_", " "),
                wordnet_definition=row["definition"],
            )
            response = completion_with_backoff(gpt_prompt)
            simplified_definition = response["choices"][0]["text"].strip().rstrip(".")
            output_writer.writerow(
                [
                    row["key"],
                    row["lemma"],
                    row["pos"],
                    simplified_definition,
                    row["definition"],
                ]
            )
            already_simplified.add(row["key"])
            f.flush()


if __name__ == "__main__":
    main()

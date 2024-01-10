#!/usr/bin/python3

import csv
import os

import openai

openai.api_key = "[REDACTED]"

FIELDNAMES = ["name", "slug"]


def get_answer(institution_name):
    if "Archive" in institution_name:
        return "NO"

    if "Library" in institution_name or "Libraries" in institution_name:
        return "NO"

    if "Private Collection" in institution_name:
        return "NO"

    if "Museum" in institution_name:
        return "YES"

    if "Gallery" in institution_name:
        return "YES"

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt='Answer YES if "{}" is a gallery or museum. Answer NO otherwise.'.format(
            institution_name
        ),
        temperature=0,
        max_tokens=4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    answer = response.get("choices")[0].text.upper().strip()

    return answer


def main():
    in_f = open("orgs.csv", "r")
    csv_reader = csv.DictReader(in_f)

    out_f = open("filtered_orgs.csv", "w", encoding="utf-8")
    csv_writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, lineterminator="\n")
    csv_writer.writeheader()

    for row in csv_reader:
        name = row.get("name")
        answer = get_answer(name)
        print(name, answer)

        if answer == "YES":
            csv_writer.writerow(row)

    in_f.close()
    out_f.close()


if __name__ == "__main__":
    main()

import openai
import pandas as pd
import os
import config
import click
import csv

# Set up your OpenAI API credentials

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion(prompt, model=config.MODEL):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=config.TEMPERATURE,
    )
    return response.choices[0].message["content"]


def create_prompt(wordlist):
    prompt = f"Here is a list of words:{wordlist}"
    sentence_count = config.SENTENCES_PER_WORD
    prompt += f"For each of these words, create {sentence_count} sentences using the word for a grade 3 student."
    prompt += f"Draft the response as a comma delimited list that can be saved as a csv file. Column 0 is the word, Column 1 through {sentence_count} "
    prompt += f"are the {sentence_count} sentences generated for that word."
    prompt += f"Add a new line to separate the words"
    return prompt


def read_wordlist(filename):
    # Read the word list from the CSV file
    data = pd.read_csv(filename, header=None, names=["word"])
    wordlist = ",".join(data["word"].tolist())
    print(wordlist)
    return wordlist


def save_wordlist(out_filename, wordlist):
    # Check if the file exists
    if not os.path.isfile(out_filename):
        click.echo(f"The file '{out_filename}' does not exist. Creating a new file.")
        open(out_filename, "a").close()

    # Save the word list to a CSV file
    rows = wordlist.split("\n")
    with open(out_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "word",
                "sentence_1",
                "sentence_02",
                "sentence_03",
                "sentence_04",
                "sentence_05",
            ]
        )
        for row in rows:
            writer.writerow(row.split(","))
    click.echo(f"Saved the word list to '{out_filename}'")


@click.command()
@click.argument("filename", type=click.Path(exists=True))
def main(filename):
    wordlist = read_wordlist(filename)
    prompt = create_prompt(wordlist)
    response: str = get_completion(prompt)
    out_filename = "out_" + filename
    save_wordlist(out_filename, response)


if __name__ == "__main__":
    main()

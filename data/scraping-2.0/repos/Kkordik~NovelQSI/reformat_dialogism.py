import csv
import ast
from transformers import AutoTokenizer
import nltk
import json
from openai import OpenAI
import re
nltk.download('punkt')


openai_key = 'OPENAI_API_KEY'


def gpt_generate_summary(prev_summary, context):
    """
    Generate summary using GPT-3
    :param prev_summary: The previous summary, that explains what happened before the original text, if it is empty, then it is the beginning of the novel
    :param context: The part of the original text, that explains what is going on after the summary
    :return: The updated summary
    """

    client = OpenAI(api_key=openai_key)

    prompt = f"""
Please update the summary. The summary must be short and include only the most important, in the updated summary explain what happened before the original text and what happend in the original text:
Previous summary, that explains what happened before the original text, if it is empty, then it is the beginning of the novel:

"{prev_summary}"

The part of the original text, that explains what is going on after the summary:

"{context}"

Return in the format:
The John and Anna opened a small business. They were selling flowers...
"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Delete extra spaces and newline chars
    summary = re.sub(r'\s+', ' ', completion.choices[0].message.content)

    return summary


def get_characters(character_info_file):
    """
    Get the list of characters and their description from the CSV file
    :param character_info_file: The path to the CSV file with the characters info
    :return: The list of characters and str with their description
    """

    names_list = []
    characters_description = ""
    with open(character_info_file, mode='r') as file:
        # Create a CSV reader object that interprets the first row as column headers
        csv_reader = csv.DictReader(file)
        genders = {"M": "Male", "F": "Female", "X": "Non-binary", "U": "Unknown"}
        for row in csv_reader:
            names_list.append(row['Main Name'])
            characters_description += f"{row['Main Name']}: Aliases: {row['Aliases']}. Gender: {genders.get(row['Gender'], 'Unknown')}. The character is: {row['Category']}\n"

    return names_list, characters_description


def cut_context(quote_byte_spans, one_way_context_length, opened_text_file):
    """
    Cut the context before and after the quote, so that the quote is in the middle of the context.
    :param quote_byte_spans: The byte spans of the quote in the text. As a list of two integers [start, finish]
    :param one_way_context_length: The length of the context before and after the quote, not including the quote itself
    :param opened_text_file: The opened text file with the novel
    :return: The start and finish byte spans of the context, the context text before and after the quote
    """
    # Get the context before the quote
    if quote_byte_spans[0] <= one_way_context_length + 1:
        context_start = 0
        opened_text_file.seek(context_start)
        before_context = opened_text_file.read(quote_byte_spans[0] - 1)
    else:
        context_start = quote_byte_spans[0] - 1 - one_way_context_length
        opened_text_file.seek(context_start)
        before_context = opened_text_file.read(one_way_context_length)

    # Delete the first sentence from the context, to avoid cutting the sentence in the middle
    nltk_before_sent = nltk.sent_tokenize(before_context)
    deleted_first_sent_len = len(nltk_before_sent.pop(0))
    before_context = before_context[deleted_first_sent_len:]
    context_start += deleted_first_sent_len

    # Get the context after the quote
    opened_text_file.seek(quote_byte_spans[1] + 1)
    after_context = opened_text_file.read(one_way_context_length)

    # Delete the last sentence from the context, to avoid cutting the sentence in the middle
    nltk_after_sent = nltk.sent_tokenize(after_context)
    deleted_last_sent_len = len(nltk_after_sent.pop(-1))
    after_context = after_context[:-deleted_last_sent_len]
    context_finish = quote_byte_spans[1] + 1 + one_way_context_length - deleted_last_sent_len

    return context_start, context_finish, before_context, after_context


def reformat_dialogism_dataset(tokenizer, context_length, book_dataset_dir):
    """
    Reformat the dialogism dataset to the format of the QA dataset for BERT based models (Context is split on before context and after context)
    :param tokenizer: The tokenizer to use for counting the tokens
    :param context_length: The length of the context in bytes
    :param book_dataset_dir: The path to the directory with the book dataset
    :return: None. The reformatted dataset is saved in the same directory and named "full_qa_data.jsonl"
    """
    current_summary = {"text": "", "covers_to": 0}

    names_list, characters_description = get_characters(f'{book_dataset_dir}/character_info.csv')

    # Open the CSV file in read mode
    with open(f'{book_dataset_dir}/quotation_info.csv', mode='r') as file:
        # Create a CSV reader object that interprets the first row as column headers
        csv_reader = csv.DictReader(file)

        with open(f'{book_dataset_dir}/novel_text.txt', mode='r') as text_file:
            # Access data by column names
            for row in csv_reader:
                quote_byte_spans_list = ast.literal_eval(row["quoteByteSpans"])
                quotes_list = ast.literal_eval(row["subQuotationList"])

                for i, quote in enumerate(quotes_list):
                    quote_byte_spans = quote_byte_spans_list[i]
                    one_way_context_length = int((context_length - len(quote)) / 2)

                    context_start, context_finish, before_context, after_context = cut_context(quote_byte_spans, one_way_context_length, text_file)

                    # Update the current summary and future summary and char list
                    if context_start >= current_summary["covers_to"]:
                        text_file.seek(current_summary["covers_to"])
                        total_context = f'{text_file.read(context_start-current_summary["covers_to"])}{before_context}"{quote}{after_context}'

                        current_summary["text"] = gpt_generate_summary(
                            current_summary["text"],
                            total_context
                        )
                        current_summary["covers_to"] = context_finish

                    question = f"Who said '{quote}' ?"
                    res = {
                        "summary": current_summary["text"],
                        "characters": characters_description,
                        "before_context": before_context,
                        "after_context": after_context,
                        "quote": quote,
                        "question": question,
                        "answer": row["speaker"],
                        "tokens_len": len(tokenizer.tokenize(f"{characters_description} {current_summary['text']} {before_context} {quote} {after_context} {question}"))
                    }

                    with open(f'{book_dataset_dir}/full_qa_data.jsonl', 'a') as outfile:
                        line = json.dumps(res)
                        outfile.write(line + '\n')


def reformat_data(reformat_file, reformated_file):
    """
    Reformat the data to the format of the QA dataset for BERT based models
    (Context is full for question answering and include: char list, summary, before+quote+after context)
    :param reformat_file: The path to the file with the data to reformat
    :param reformated_file: The path to the file to save the reformatted data
    :return: None. The reformatted dataset is saved in the same directory and named "qa_data_reformatted.jsonl"
    """
    i = 0
    with open(reformat_file, mode='r') as file:
        with open(reformated_file, 'w') as outfile:
            for line in file:
                data = json.loads(line)
                context = f'Characters:\n{data["characters"]}\nSummary:\n{data["summary"]}\nNovel Text:\n{data["before_context"]}"{data["quote"]}"{data["after_context"]}'
                answer_start = context.lower().find(data["answer"].lower())

                new_data = {
                    "context": context,
                    "question": f'Which character said "{data["quote"]}"?',
                    "answers": {"answer_start": [answer_start], "text": [data["answer"]]},
                    "id": f"question-{i}",
                }
                line = json.dumps(new_data)
                outfile.write(line + '\n')
                i += 1


if __name__ == "__main__":
    book_dataset_dir_name = "TheGambler"
    context_length = 4500
    tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
    reformat_dialogism_dataset(tokenizer, book_dataset_dir_name)
    reformat_data(f"{book_dataset_dir_name}/full_qa_data.jsonl", f"{book_dataset_dir_name}/qa_data_reformatted.jsonl")

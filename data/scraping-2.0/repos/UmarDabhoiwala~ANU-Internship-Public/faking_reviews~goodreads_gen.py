import csv
import openai
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import re
import math
from halo import Halo
from faker import Faker
import random
from datetime import datetime

fake = Faker()
openai.api_key = config.OPENAI_API_KEY
spinner = Halo(text='Loading', spinner='dots')

def chat_gpt_completion(chat_message, append = False, usr_prompt = ""):

    spinner.start("text generating")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= chat_message
    )

    spinner.succeed("text generated")

    response = completion['choices'][0]["message"]["content"]

    if append:
        chat_message.append({"role": "system", "content": response})

    return response, chat_message


def write_review(book_title, book_author):
    prompt = f"""
    Write a short (less than 100 words) funny book review for the book {book_title} by {book_author}.
    Start the review with the rating you give the book out of 5 enclosed in square braces for example [3]:review.
    """

    review, _ = chat_gpt_completion([{"role": "user", "content": prompt}])

    rating, review = rating_and_review(review)

    return rating, review



def rating_and_review(text):
    number = re.search(r"\[(\d+)\]", text)
    if number:
        extracted_number = number.group(1)
        print(f"Extracted number: {extracted_number}")

        modified_text = re.sub(r"\[\d+\]", "", text).strip()
        print(f"Modified text: {modified_text}")

        return extracted_number, modified_text
    else:
        return None, text


def write_review_csv(filename, book_title, book_author, isbn, rating, read_date, publisher, publication_date, review):
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Title', 'Author', 'ISBN', 'My Rating', 'Publisher', 'Year Published', 'Date Read', 'My Review']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Author', 'ISBN', 'My Rating', 'Publisher', 'Year Published', 'Date Read', 'My Review']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'Title': book_title,
            'Author': book_author,
            'ISBN': isbn,
            'My Rating': rating,
            'Publisher': publisher,
            'Year Published': publication_date,
            'Date Read': read_date,
            'My Review': review,
        })




def get_top_books(stop, file_name = "books.csv"):
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        header = next(csvreader)
        list_books = []
        for i, row in enumerate(csvreader):
            title, authors, average_rating, isbn, isbn13,publication_date, publisher = row [:7]
            book_thing = [title, authors, average_rating, isbn, isbn13, publication_date, publisher]
            print (f"Title:{title}")
            list_books.append(book_thing)
            if i == stop:
                break

        return list_books


def writeTopBooks(num):

    thing = get_top_books(num)

    for x in thing:
        title = x[0]
        authors = x[1]
        avg_rat = x[2]
        isbn = x[3]
        publication_date = x[5]
        publisher = x[6]
        rate, review = write_review(title, authors)
        if rate == None:
            rate = str (math.floor(float(avg_rat)))

        date = fake.date_between(start_date='-5y', end_date='today')
        date = date.strftime("%Y-%m-%d")

        date_object = datetime.strptime(publication_date, "%m/%d/%Y")
        formatted_publication_date = date_object.strftime("%Y-%m-%d")
        print(formatted_publication_date)
        write_review_csv("book_reviews.csv", title, authors, isbn, rate, date, publisher, formatted_publication_date, review)


def shuffle_csv (file_name):
    with open(file_name, 'r', encoding='utf-8') as infile:
        csvreader = csv.reader(infile)

        # Read the header row
        header = next(csvreader)

        # Read the rest of the rows
        rows = list(csvreader)

    # Shuffle the rows
    random.shuffle(rows)

    # Write the shuffled rows to a new CSV file
    with open(file_name, 'w', encoding='utf-8', newline='') as outfile:
        csvwriter = csv.writer(outfile)

        # Write the header row
        csvwriter.writerow(header)

        # Write the shuffled rows
        for row in rows:
            csvwriter.writerow(row)

def generate_book_reviews(num_review):
    shuffle_csv("books.csv")

    source_file = 'book_reviews.csv'
    destination_folder = 'files/book_reviews.csv'  # Replace with the actual path to the destination folder

    writeTopBooks(num_review)

    # Check if the source file exists
    if os.path.exists(source_file):
        # Check if the destination file exists
        destination_file = os.path.join(destination_folder, os.path.basename(source_file))
        if os.path.exists(destination_file):
            # Remove the existing destination file
            os.remove(destination_file)

        # Move the source file to the destination folder
        shutil.move(source_file, destination_folder)
        print("File moved successfully.")
    else:
        print("Source file not found.")

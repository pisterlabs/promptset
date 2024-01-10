import os
import json
import re
import fitz  # PyMuPDF
import openai
import time
from difflib import SequenceMatcher


def load_questions_and_answers(filepath):
    """
    Load the questions and answers from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary contains a question and answer.
    """
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r") as f:
            qa_list = json.load(f)
    except json.decoder.JSONDecodeError:
        print(f"Error: Empty or invalid JSON file '{filename}'. Skipping file.")
        qa_list = []
    return qa_list


def extract_text_from_page(pdf_file, page_number):
    """
    Extract the text from a page in a PDF file.

    Args:
        pdf_file (fitz.Document): The PDF file.
        page_number (int): The page number.

    Returns:
        The text on the page.
    """
    page = pdf_file[page_number]
    page_text = page.get_text()
    return page_text


def generate_answer(question, page_text):
    # Clean the page text by removing extra spaces
    page_text = re.sub(r'\s+', ' ', page_text)

    """
    Use OpenAI's GPT-3 to generate an answer to a question based on the text on a page.

    Args:
        question (str): The question.
        page_text (str): The text on the page.

    Returns:
        The generated answer.
    """
    # Define the conversation history as a list of messages
    conversation = [
        {"role": "system", "content": f"please generate an answer for the following'{question}' based on the page text only."},
        {"role": "user", "content": page_text},
    ]
    # Use OpenAI's GPT-3 to generate a prompt and completion
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                max_tokens=96,
                temperature=0.1  # Adjust this value to control the randomness of the output
            )
            break
        except (openai.error.ServiceUnavailableError, openai.error.APIError,
                openai.error.APIConnectionError, openai.error.InvalidRequestError) as e:
            print(f"OpenAI API error: {e}")
            if "This model's maximum context" in str(e):
                print("skip...moving to next page")
                break # skip this page
            else:
                print("Retrying in 1 minute...")
                time.sleep(60)
    if completion:
        # Extract the prompt and completion from the response
        response = completion.choices[0].message["content"]
        response = response.replace("\n", "")
        return response
    else:
        return None


def check_answer(answer, generated_answer):
    """
    Check if the generated answer is similar to the answer.

    Args:
        answer (str): The original answer.
        generated_answer (str): The generated answer.

    Returns:
        True if the generated answer is similar to the answer, False otherwise.
    """
    # Split the answer and generated answer strings into lists of words
    answer_words = answer.split()
    generated_answer_words = generated_answer.split()
    # Check if most of the words in the answer are also part of the generated answer
    num_common_words = 0
    for word in answer_words:
        if word in generated_answer_words:
            num_common_words += 1
    if num_common_words >= len(answer_words) * 0.8:
        print("The generated answer is similar to the answer.")
        return True
    else:
        print("The generated answer is not similar to the answer.")
        return False


def rewrite_answer(answer, filepath, question):
    """
    Rewrite an answer based on the text on a page.

    Args:
        answer (str): The original answer.
        filepath (str): The path to the JSON file containing the question and answer.
        question (str): The question to rewrite the answer for.

    Returns:
        None.
    """
    with open(filepath, "r") as f:
        qa_list = json.load(f)
    for qa_dict in qa_list:
        if qa_dict.get("prompt") == question:
            qa_dict["completion"] = answer
            break
    with open(filepath, "w") as f:
        json.dump(qa_list, f, indent=4)


def process_file(filepath, pdf_file):
    """
    Process a JSON file containing questions and answers.

    Args:
        filepath (str): The path to the JSON file.
        pdf_file (fitz.Document): The PDF file.

    Returns:
        None.
    """
    try:
        # Load the questions and answers from the JSON file
        qa_list = load_questions_and_answers(filepath)
        # Extract the page number from the filename
        page_number = int(os.path.basename(filepath).split("_")[2])
        # Iterate through the questions and answers
        for qa_dict in qa_list:
            # Extract the question and answer from the dictionary
            question = qa_dict.get("prompt")
            answer = qa_dict.get("completion")
            # Extract the text from the page
            page_text = extract_text_from_page(pdf_file, page_number)
            # Generate an answer to the question based on the text on the page
            generated_answer = generate_answer(question, page_text)
            # Check if the generated answer is correct according to the page text
            if generated_answer and not check_answer(answer, generated_answer):
                # Rewrite the answer based on the text in the PDF file
                rewrite_answer(generated_answer, filepath, question)
        # Rename the file to include "checked" in the filename
        os.rename(filepath, filepath.replace(".json", "_checked.json"))
        print(f"Finished processing '{filepath}'.")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def main():
    # Define the directory containing the JSON files
    directory = "/home/erangross/MedicalChatGPT/datasets/Braunwald-heart-disease"
    # Load the PDF file
    pdf_file_name = "/home/erangross/MedicalChatGPT/books/Braunwald-heart-disease-done.pdf"
    pdf_file = fitz.open(pdf_file_name)
    # Iterate through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json") and "checked" not in filename:
            filepath = os.path.join(directory, filename)
            process_file(filepath, pdf_file)


if __name__ == "__main__":
    main()
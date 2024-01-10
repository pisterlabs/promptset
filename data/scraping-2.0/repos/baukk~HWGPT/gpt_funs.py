import os
import openai
import subprocess
import preprocess

# Set your OpenAI API key


# def preprocess(file_path):
openai.api_key = "Enter your ChatGPT API key here"


# Reads one text file and processes it with chat gpt and documents and answers the questions and returns one text file back
def gpt_format_page(page_path):
    with open(page_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": ''' Assist in rewriting text extracted from scanned question papers or homework 
                assignments. The provided text may lack proper formatting or contain errors due to the scanning process. 
                Your task is to utilize your knowledge to rewrite the text in a clear and understandable manner. You have 
                the freedom to correct errors, add missing words, and ensure the questions are properly formatted as you 
                would in a well-documented question paper. Write answers for every Question below it starting with 
                keyword 'Answer'. '''
            },
            {
                "role": "user",
                "content": "Take the following data and rewrite it as instructed " + file_content
            }
        ]
    )
    processed_text = response['choices'][0]['message']['content']

    file_name = os.path.basename(page_path)
    os.makedirs("Solved_output_folder", exist_ok=True)
    output_file = f"Solved_output_folder/solved_{file_name}"  # Generate the output file name
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    print("Text extracted and saved to", output_file)
    return processed_text

#takes a folder with text files and merges all text files to one text file.

def merge_into_one_text_file(folder_path):
    output_file = f'{folder_path}/merged.txt'
    input_files = os.listdir(folder_path)
    print(input_files, "input files")
    input_files = [f"{folder_path}/"+ file_name for file_name in input_files]
    print(input_files)

    with open(output_file, 'w', encoding='utf-8') as output:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as input_file:
                output.write("Page # {}\n".format(input_files.index(file) + 1))
                output.write(input_file.read())
                output.write("\n\n")

    print("Text files merged into", output_file)
    return output_file

# recieves the folder created by pdf2folder(pdf to text of all pages into one folder) and processes each page with gpt_format_page and saves the output in a new folder
def format_entire_pdf(folder_path):
    print("start")

    output_file = f"{folder_path}/ansmerged.txt"
    input_files = os.listdir(folder_path)
    # input_files.remove("merged.txt")
    input_files.remove("merged.txt")
    input_files = [f"{folder_path}/" + file_name for file_name in input_files]
    with open(output_file, 'w', encoding='utf-8') as output:
        print(enumerate(input_files))
        print("in")
        print("hellocheckennumerate")
        for i, file in enumerate(input_files):
            processed_text = gpt_format_page(f"{folder_path}/{folder_path.split('/')[-1]}.pdf_page{i + 1}.txt")
            output.write("Page # {}\n".format(i + 1))
            output.write(processed_text + "\n\n")
            print("Page # {} processed.".format(i + 1))
    print("Text extracted and saved to", output_file)
def test():
    print(2+2)
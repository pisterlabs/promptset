import requests
import time
import os
import csv
from pdfminer.high_level import extract_text
import pdfplumber
import fitz


from openai import OpenAI

client = OpenAI() #api_key=OPENAI_API_KEY)

def main():
    pmid_to_path = {}
    #get_pmid_to_paths("oa_file_list.csv", pmid_to_path)
    #get_pmid_to_paths("oa_non_comm_use_pdf.csv", pmid_to_path)
    print("map done")

    directory_path = 'data/'
    process_pdfs_in_directory(directory_path)

def process_pdfs_in_directory(directory):
    i = 0
    f = open('output.txt', 'w')

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pmid = filename.split('.')[0]
            print(i, filename, pmid)
            pdf_path = os.path.join(directory, filename)
            #text = extract_text(pdf_path)
            text = get_text(pdf_path)
            result = process(pmid, text)
            f.write(result + '\n')
            f.flush()

        i += 1

def get_text(pdf_path):
    text = ''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # Extract text in dict format
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if 'lines' in block:  # Ensure it's a text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + ' '
                    text += '\n'
    return text


def get_text2(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            # Extract text from the page while maintaining the layout
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def get_pmid_to_paths(file_name, pmid_to_path):
    print(f"getting paths for {file_name}")
    f = open(file_name)
    with open(file_name) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            try:
                path = line[0]
                pmid = line[4]
                #print(path, pmid)
            except Exception as e:
                print(e)
                continue

            if len(pmid) == 0:
                continue

            if pmid in pmid_to_path:
                print("uhhhh", pmid, path)
                continue

            pmid_to_path[pmid] = path
     
def process(pmid, content):
    print("processing", pmid)
    base_prompt = """Given the following research paper about mouse models for cardiology conditions, please extract the following fields to JSON:
        {
            "Study Name": text, // Use study name from the initial response when possible.
            "Year Published": integer,
            "Mouse Type": text // Mouse Type options "ApoE KO", "LDLr KO",
            "Text used to identify sex of mice": text // VERBATIM text used to determine "sex of mice" output. Must reference "mice" explicitly.
            "Sex of mice": text // Sex of mice explicitly referenced: "Male", "Female" calculated from "text used to identify sex of mice".
            "Impacted Genete": text,
            "Gene Symbol": text,
            "Vessel Location": text // eg. Aorta or aortic branch,
            "Lesion Size": text // options: Increase, Decrease, Neutral,
            "Plaque Inflamation": text // options: Increase, Decrease, Neutral,
            "Change in Lipid Content": text options: Increase, Decrease, Neutral,
            "Pubmed ID": int
        }

        IGNORE sex from human patients.
        --------------- 
    """
    context_window = 16000
    context_window = 60000
    for i in range(0, len(content), context_window):
        if i == 0:
            print(i, i+context_window)
            prompt = base_prompt + "The content is below for pub med id " + pmid + ": " + content[i:i+context_window]
            response = open_ai_request(prompt)
            last_content = response.choices[0].message.content
            print(last_content)
        else:
            print(i, i+context_window)
            prompt = base_prompt + f"An initial answer was {last_content}. You can update with the following info: " + content[i:i+context_window]
            response = open_ai_request(prompt)
            last_content = response.choices[0].message.content
            print(last_content)

    return last_content

def open_ai_request(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", #gpt-3.5-turbo-1106",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt + " ONLY USE CONTENT PROVIDED. ONLY OUTPUT JSON",
                }
            ]
        )
        print(response)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    main()

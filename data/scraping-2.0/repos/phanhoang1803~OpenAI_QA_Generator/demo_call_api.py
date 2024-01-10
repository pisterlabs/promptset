def pdf_load_text(file_path):
    # Load data from disk
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    no_documents = len(docs)
    merged_content = ""
    # Loop through each page and merge the content to text instead of docs
    for i in range(no_documents):
        merged_content += docs[i].page_content + '\n'

    # print(merged_content)
    return merged_content

def get_mcqs(saved_path = "mcq_result.json"):
    # Call API to get response
    BASE_URL = 'http://localhost:5000'
    response = requests.post(f'{BASE_URL}/mcq', json={'text': merged_content})

    print("\n----------------Generating questions----------------\n")
    if response.status_code == 200:
        result = response.json()
        
        # Save json file
        with open(saved_path, 'w') as json_file:
            json.dump(result, json_file, indent=2)
        
        return True
    else:
        print(f"\nRequest failed with status code: {response.status_code}")
        print(response.text)
        
        return False

##-------------------------------------------------##

import requests
import json
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

merged_content = pdf_load_text(file_path="Lecture.Writing.pdf")

get_mcqs(saved_path="multil_choice_questions_LectureWriting.json")
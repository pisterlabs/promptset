import openai
import os
import re
from dotenv import load_dotenv
from multiprocessing import Process
from PyPDF2 import PdfReader
import csv
import requests

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Lecture:
    def __init__(self, class_name, lec_name, url):
        self.class_name = class_name
        self.lec_name = lec_name
        self.url = url

def get_txt(source):
    reader = PdfReader(source)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_config(filename):
    configs = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            configs.append(Lecture(*row))

    return configs

def convert(filename):
    # base_url = "https://mau-md--embeddings-app.modal.run/kill"
    # requests.get(base_url)

    configs = read_config(filename)

    for lecture in configs:
        print(lecture.lec_name)
        response_text = get_txt(f'lecture_materials/{lecture.lec_name}.pdf')
        with open(f'lecture_converted/{lecture.lec_name}.txt', 'w') as file:
            file.write(response_text)    

    for lecture in configs:
        print(lecture.url)
        with open(f'lecture_converted/{lecture.lec_name}.txt', 'r') as file:
            contents = file.read().split()
            contents = ' '.join(contents)
            contents = re.split('\.|\!|\?', contents)

        # Add class
        base_url = "https://mau-md--embeddings-app.modal.run/add-class"
        params = {
            "class_name": lecture.class_name,
        }
        response = requests.post(base_url, json=params)
        class_id = response.text

        # Add Lecture
        base_url = "https://mau-md--embeddings-app.modal.run/add-lecture"
        params = {
            "lecture_name": lecture.lec_name,
            "class_id": class_id,
            "url": lecture.url,
        }
        response = requests.post(base_url, json=params)
        lec_id = response.text

        i = 0
        sz = 15
        categories = {}
        chunk = []
        print(lecture.class_name, lecture.lec_name, lec_id)
        while i*sz < len(contents):
            base_url = "https://mau-md--embeddings-app.modal.run/add-context"
            params = {
                # "lecture_id": lecture.lec_id,
                "lecture_id": lec_id,
                "query": ". ".join(contents[i*sz:(i+2)*sz]),
                "lecture_url": lecture.url,
            }
            response = requests.post(base_url, json=params)
            print(response)
            i += 1
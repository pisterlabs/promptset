from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader 
from json.decoder import JSONDecodeError
import google.generativeai as palm
import pprint
import json

def load_json_template(template_path):
    try:
        with open(template_path, 'r') as f:
            json_template = json.load(f)
    except FileNotFoundError as e:
        print(f"Your json template is not found. Please enter the correct file path.\n"
          f"{e}")
    except JSONDecodeError as e:
        print(f"Your json template is not well formated. Please check the format of the json file.\n"
          f"{e}")
    json_template_string = json.dumps(json_template)
    #print(ResumeTemplateString)
    return json_template_string

def load_resume(resume_path):
    try:
        reader = PdfReader(resume_path)
    except FileNotFoundError as e:
        print(f"Your resume is not found. Please enter the correct file path.\n"
          f"{e}")
    page = reader.pages[0] 
    resume_text = page.extract_text() 
    return resume_text

def call_palm_api(json_template_string, resume_text, palm_api_key):
    palm.configure(api_key = palm_api_key)
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[1].name
    print(model)
    prompt = "Extract text of cv and organize it into json format like this : " + json_template_string + "\n" + resume_text
    completion_palm = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        max_output_tokens=10000,
    )
    print(completion_palm.result)
    return completion_palm.result
    
def call_openal_api(json_template_string, resume_text, openai_api_key):
    client = OpenAI(
        api_key = openai_api_key,
    ) 
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Extract text of cv and organize it into json format like this : " + json_template_string + "\n" + resume_text,
            },
        ],
    )
    return completion.choices[0].message.content

def convert_text_to_json(generated_text):
    try:
        output_object = json.loads(generated_text)
    except JSONDecodeError as e:
        print(f"The output cannot be formated into json. Please check the input resume and output json.\n"
          f"{e}")
    return output_object

def output_json_file(output_dir, user_id, output_object):
    output_file = output_dir + "ResumeParsed_" + user_id + ".json"
    try:
        with open(output_file, "w", encoding='utf-8') as of:
            json.dump(output_object, of, ensure_ascii = False, indent = 4)
    except FileNotFoundError as e:
        print(f"The output file cannot be created. Please check the output path.\n"
          f"{e}")
    return output_file

def parse_resume(resume_path, template_path, user_id, output_dir, api_key):
    json_template_string = load_json_template(template_path)
    resume_text = load_resume(resume_path)
    generated_text = call_openal_api(json_template_string, resume_text, api_key)
    json_object = convert_text_to_json(generated_text)
    output_file = output_json_file(output_dir, user_id, json_object)
    return output_file

resume_path = "./Materials/Resume.pdf"
template_path = "./Materials/ResumeTemplate.json"
user_id = "hack"
output_dir = "./Materials/"
api_key = "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"
#resumeParse(resume_path, template_path, user_id, output_dir)
parse_resume(resume_path, template_path, user_id, output_dir, api_key)

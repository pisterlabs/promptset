from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader 
from json.decoder import JSONDecodeError
import google.generativeai as palm
import pprint
import json
import time
import threading
from multiprocessing.pool import ThreadPool

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
    #print(json_template_string)
    return json_template_string

def load_resume(resume_path):
    try:
        reader = PdfReader(resume_path)
    except FileNotFoundError as e:
        print(f"Your resume is not found. Please enter the correct file path.\n"
          f"{e}")
    page = reader.pages[0] 
    resume_text = page.extract_text() 
    print(resume_text)
    return resume_text

def call_palm_api_once(resume_text, template, model):
    prompt = "Extract the content of the resume:\n" + resume_text + "\nand fill them into current json template: " + template+ "\n" + "Only the content in the current json template provided is needed!!! Please don't extract the content needed by prior json template! Also make sure that the Highlights section is of length 75 words maximum"
    completion_palm = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        max_output_tokens=1000,
    )
    print(completion_palm.result)
    json_text = completion_palm.result
    json_object = json.loads(json_text)
    return json_object
    

def call_palm_api(resume_text, palm_api_key):
    palm.configure(api_key = palm_api_key)
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name
    print(model)
    template1 = load_json_template("../Materials/ResumeTemplate1.json")
    template2 = load_json_template("../Materials/ResumeTemplate2.json")
    template3 = load_json_template("../Materials/ResumeTemplate3.json")
    template4 = load_json_template("../Materials/ResumeTemplate4.json")
    # prompt = "Extract text of the resume and organize it into json format like this : " + json_template_string + "\n" + resume_text
    # prompt = "Organize the content of the resume:\n" + resume_text + "\ninto json format like this : " + json_template_string + "\n"
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=10000,
    # )
    # t1 = threading.Thread(target=call_palm_api_once, args=(resume_text, template1, model, ))
    # t2 = threading.Thread(target=call_palm_api_once, args=(resume_text, template2, model, ))
    # t3 = threading.Thread(target=call_palm_api_once, args=(resume_text, template3, model, ))
    # t4 = threading.Thread(target=call_palm_api_once, args=(resume_text, template4, model, ))

    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()

    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()


    pool = ThreadPool(processes=4)
    async_result_1 = pool.apply_async(call_palm_api_once, (resume_text, template1, model))
    async_result_2 = pool.apply_async(call_palm_api_once, (resume_text, template2, model))
    async_result_3 = pool.apply_async(call_palm_api_once, (resume_text, template3, model))
    async_result_4 = pool.apply_async(call_palm_api_once, (resume_text, template4, model))
    json_object_1 = async_result_1.get() 
    json_object_2 = async_result_2.get() 
    json_object_3 = async_result_3.get() 
    json_object_4 = async_result_4.get() 

    # prompt = "Organize the content of the resume:\n" + resume_text + "\naccording to json format : " + template1 + "\n" + "Only the content in the json is needed!!!"
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=1000,
    # )
    # #print(completion_palm.result)
    # json_text_1 = completion_palm.result
    # json_object_1 = json.loads(json_text_1)
    # prompt = "Organize the content of the resume:\n" + resume_text + "\naccording to json format : " + template2 + "\n" + "Only the content in the json is needed!!!"
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=1000,
    # )
    # #print(completion_palm.result)
    # json_text_2 = completion_palm.result
    # json_object_2 = json.loads(json_text_2)
    # prompt = "Organize the content of the resume:\n" + resume_text + "\naccording to json format without any other extra text: " + template3 + "\n" + "Only the content in the json is needed!!!"
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=1000,
    # )
    # print(completion_palm.result)
    # json_text_3 = completion_palm.result
    # json_object_3 = json.loads(json_text_3)
    # prompt = "Organize the content of the resume:\n" + resume_text + "\naccording to json format without any other extra text: " + template4 + "\n" + "Only the content in the json is needed!!!"
    # completion_palm = palm.generate_text(
    #     model=model,
    #     prompt=prompt,
    #     temperature=0,
    #     max_output_tokens=1000,
    # )
    # #print(completion_palm.result)
    # json_text_4 = completion_palm.result
    # json_object_4 = json.loads(json_text_4)

    for key, value in json_object_2.items():
        json_object_1[key] = value
    
    for key, value in json_object_3.items():
        json_object_1[key] = value

    for key, value in json_object_4.items():
        json_object_1[key] = value

    return json.dumps(json_object_1)
    
def call_openal_api(resume_text, openai_api_key):
    client = OpenAI(
        api_key = openai_api_key,
    ) 

    print("Loading...")
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
    generated_text = call_palm_api(resume_text, "AIzaSyB1Y9HF5GBUAKXBa-d3-1G_IxKKU93cFI8")
    # generated_text = call_openal_api(json_template_string, resume_text, api_key)
    #json_object = convert_text_to_json(generated_text)
    #output_file = output_json_file(output_dir, user_id, json_object)
    return generated_text

# resume_path = "./Materials/Resume.pdf"
# template_path = "./Materials/ResumeTemplate.json"
# user_id = "hack"
# output_dir = "./Materials/"
# api_key = "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"
# #resumeParse(resume_path, template_path, user_id, output_dir)
# start = time.time()
# parse_resume(resume_path, template_path, user_id, output_dir, api_key)
# end = time.time()
# print(end - start)

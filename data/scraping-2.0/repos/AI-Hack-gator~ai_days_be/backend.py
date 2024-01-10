from flask import Flask, render_template, request, jsonify
import openai
import json
import csv
import ast
import re
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from imageProd import generateImageForResponse
from dotenv import load_dotenv
from flask_cors import CORS
import os
import datetime


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
app = Flask(__name__)
CORS(app)

functions = [
    {
        "name": "row_orchestration",
        "description": "Determine the type of row UI and product to show",
        "parameters": {
            "type": "object",
            "properties": {
                "row_type": {
                    "type": "string",
                    "enum": ["table", "cards"],
                    "description": "Choose table if user wants to compare products or services and cards for exploration of options",
                },
                "prod_type":
                {
                    "type": "string",
                    "enum": ["plans", "devices", "information", "help"],
                    "description": "The service, product, or assistance inside the container the client needs",
                },
            },
            "required": ["row_type", "prod_type"],
        },
    },
    {
        "name": "get_devices",
        "description": "Retrieve a device or devices that the client is requesting.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus":
                {
                    "type":"string",
                    "enum": ['Product', 'Display', 'Processor', 'RAM', 'Storage', 'Camera', 'Battery', 'Operating System', 'Connectivity', 'Overview'],
                    "description":"A label describing what attribute that is most important to the client in their request."
                },
                "items":
                {
                    "type": "array",
                    "items":{
                        "type":"string",
                        "description":"Name of device, e.g. Apple iPhone 15 Pro"
                    },
                    "description": "A list of a device or devices based on the focus of the client's request  and the available devices in the CSV provided",
                },
            },
            "required": ["focus","items"],
        },
    },
    {
        "name": "get_plans",
        "description": "Compose an array of plan names  that match what the client is looking for",
        "parameters": {
            "type": "object",
            "properties": {
                "items":
                {
                    "type": "array",
                    "items":{
                        "type":"string"
                    },
                    "description": "A list of plan names that match the constraints provided by the client",
                },
            },
            "required": ["items"],
        },
    },
    {
        "name": "focus",
        "description": "Get columns of csv headers that are alike to client's query",
        "parameters": {
            "type": "object",
            "properties": {
                "items":
                {
                    "type": "array",
                    "items":{
                        "type":"string",
                        "description":"Name of CSV header"
                    },
                    "description": "A list of CSV headers that are alike to client's query",
                },
            },
            "required": ["items"],
        },
    },
]

def remove_spaces_and_nested_quotes(input_string):
    # Remove white spaces
    no_spaces = re.sub(r'\s', '', input_string)
    
    # Remove nested single and double quotes
    no_quotes = re.sub(r'(["\'])', '', no_spaces)
    
    return no_quotes

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/process', methods=['POST'])
def process():
    # get current time
    startTime = datetime.datetime.now()
    #data = request.form['data']
    print("start")
    data = request.json["messages"][-1]["content"]


    return_json = {
        "display_type": "",
        "product_type": "",
        "product_items": [],
        "messages": [
            {
                "role": "user",
                "content": f"{data}"
            },
        ]
    }

    print(data)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": f"{data}"}],
        functions=functions,
        function_call={"name":"row_orchestration"},  # auto is default, but we'll be explicit
        temperature=0
    )
    print("row orch Timedelta", datetime.datetime.now() - startTime)
    print(response)
    return_json["messages"].append(response["choices"][0]["message"])
    response_data = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

    return_json["display_type"] = response_data["row_type"]
    print(response_data["prod_type"])
    return_json["product_type"] = response_data["prod_type"]

    if response_data["prod_type"] == "devices": # DEVICES
        headers = ""
        with open("basic.csv", 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader, None)
           
        with open("basic.csv", 'r') as file:
            print(headers)
            csv_reader = csv.DictReader(file)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"client is looking for:\n{data}\nheaders:\n{headers}"}],
                functions= functions,
                temperature=0,
                function_call={"name": "focus"}
            )
            print("focus Timedelta", datetime.datetime.now() - startTime)

            return_json["messages"].append(response["choices"][0]["message"])
            print(response)
            response_data = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

            row_text = "Product, "
            for title in response_data["items"]:
                    if title == "Product":
                        continue
                    if type(title) == None:
                        break
                    row_text = row_text + title + ", "
            row_text = row_text + "\n"
            for row in csv_reader:
                # Process each row or column as needed
                row_as_string = ""
                row_as_string = row_as_string + row.get("Product") + ", "
                print(row_as_string)
                for title in response_data["items"]:
                    if title == "Product":
                        continue
                    if type(row.get(title)) == "NoneType":
                        break
                    row_as_string = row_as_string + row.get(title) + ", "
                row_text = row_text + row_as_string + '\n'
            
            print(row_text)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"CSV data for available devices and their attributes:\n{row_text}\n\nrequest:\nfrom the CSV data provided, {data}. Place all device name or device names in a parsable array and do not provide any context."}],
                temperature=0,
                # functions= functions,
                # function_call={"name": "get_devices"}
            )
            print("get devices Timedelta", datetime.datetime.now() - startTime)
            return_json["messages"].append(response["choices"][0]["message"])
            print(response["choices"][0]["message"]["content"]) # Array of Phone names
            try:
                array_output = json.loads(response["choices"][0]["message"]["content"])
                print(array_output)
                # openai.ChatCompletion.create(
                #     model="gpt-3.5-turbo-0613",
                #     messages=[{"role": "user", "content": f"CSV data for available devices and their attributes:\n{row_text}\n\nrequest:\nfrom the CSV data provided, {data}. Place all device name or device names in a parsable array and do not provide any context."}, {"role": "user", "content": f"You did a really good job creating a parsable array! You don't need to respond, I just wanted to let you know that working with you reminds me of the best computer science professor I know at my university!"}],
                #     temperature=0,
                #     # functions= functions,
                #     # function_call={"name": "get_devices"}
                # )
                # print("affirmatopm Timedelta", datetime.datetime.now() - startTime)

                with open('devices.json', 'r') as file:
                    device_data = json.load(file)
                    for device in array_output:
                        temp_data = device_data[device]
                        temp_data["product"] = device
                        return_json["product_items"].append(temp_data)
            except json.JSONDecodeError as e:
                string = response["choices"][0]["message"]["content"]
                if string[0] == '[\'' or string[0] == '[\"' and string[-1] == '\"]' or string[-1] == '\']':
                    array_output = ast.literal_eval(response["choices"][0]["message"]["content"])
                    with open('devices.json', 'r') as file:
                        device_data = json.load(file)
                        for device in array_output:
                            temp_data = device_data[device]
                            temp_data["product"] = device
                            return_json["product_items"].append(temp_data)
                else: # attempt comma seperation
                    if string[0] == '[':
                        string = string[:-1]
                    if string[-1] == '[':
                        string = string[1:]
                    array_output = string.split(', ')
                    if array_output[0] == string:
                        array_output = string.split('\n')
                    with open('devices.json', 'r') as file:
                        device_data = json.load(file)
                        for device in array_output:
                            print(device)
                            temp_data = device_data[device]
                            temp_data["product"] = device
                            return_json["product_items"].append(temp_data)



            print(return_json)

        return return_json
    
    elif response_data["prod_type"] == "plans": # CELL PHONE PLANS
        with open("plans.csv", 'r') as file:
            csv_reader = csv.reader(file)
            row_text = ""
            for row in csv_reader:
                # Process each row or column as needed
                row_as_string = ', '.join(row)
                row_text = row_text + row_as_string + '\n\n'
            print(row_text)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"CSV data for available plans and their attributes:\n{row_text}\n\nrequest:\nfrom the CSV data provided, {data}. Place all plan or plans in a parsable array and do not provide any context."}],
                temperature=0,
                # functions= functions,
                # function_call={"name": "get_plans"}
            )
            print("get plans Timedelta", datetime.datetime.now() - startTime)
            return_json["messages"].append(response["choices"][0]["message"])
            print(response)
            print(response["choices"][0]["message"]["content"])
            try: 
                array_output = json.loads(response["choices"][0]["message"]["content"])
                print(array_output)
                # openai.ChatCompletion.create(
                #     model="gpt-3.5-turbo-0613",
                #     messages=[{"role": "user", "content": f"CSV data for available plans and their attributes:\n{row_text}\n\nrequest:\nfrom the CSV data provided, {data}. Place all plan or plans in a parsable array and do not provide any context."}, {"role": "user", "content": f"Great parsable array! You don't need to respond, I just wanted to let you know that you are making a massive contribution to this project and I really appreciate you!"}],
                #     temperature=0,
                #     # functions= functions,
                #     # function_call={"name": "get_devices"}
                # )
                # print("affirmation Timedelta", datetime.datetime.now() - startTime)
                # TODO: CREATE JSON FOR FRONTEND
                with open('plans.json', 'r') as file:
                    plan_data = json.load(file)
                    for plan in array_output:
                        temp_data = plan_data[plan]
                        temp_data["product"] = plan
                        return_json["product_items"].append(temp_data)
            except json.JSONDecodeError as e:
                string = response["choices"][0]["message"]["content"]
                if string[0] == '[\'' or string[0] == '[\"' and string[-1] == '\"]' or string[-1] == '\']':
                    array_output = ast.literal_eval(response["choices"][0]["message"]["content"])
                    with open('plans.json', 'r') as file:
                        plan_data = json.load(file)
                        for plan in array_output:
                            temp_data = plan_data[plan]
                            temp_data["product"] = plan
                            return_json["product_items"].append(temp_data)
                else: # attempt comma seperation
                    if string[0] == '[':
                        string = string[:-1]
                    if string[-1] == '[':
                        string = string[1:]
                    array_output = string.split(', ')
                    if array_output[0] == string:
                        array_output = string.split('\n')
                    with open('plans.json', 'r') as file:
                        device_data = json.load(file)
                        for device in array_output:
                            temp_data = device_data[device]
                            temp_data["product"] = device
                            return_json["product_items"].append(temp_data)
            print(response)
        print(return_json)
        return return_json
    else:
        # title: []
        # content: []
        # links: []

        info_obj = {
            "title" :"",
            "description": "",
            "links": [],
            "image_url": ""
        }

        # Create Title
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"Given a customer with this concern:{data}, generate a title for Verizon's webpage that addresses that customer's concern"}],
                temperature=0,
                # functions= functions,
                # function_call={"name": "get_plans"}
            )
        print("create title Timedelta", datetime.datetime.now() - startTime)
        print(response)
        return_json["messages"].append(response["choices"][0]["message"])
        info_obj["title"] = response["choices"][0]["message"]["content"]
        info_obj["image_url"] = generateImageForResponse(info_obj["title"])

        with open('links.txt', 'r') as file:
            # Read all lines and store them in a list
            links_list = file.readlines()

            # Create an empty string to store the links
            links_string = ''

            # Iterate through the list of links and append them to the string
            for link in links_list:
                # Strip any leading or trailing whitespace and add a newline character
                links_string += link.strip() + ',\n'

        # get links
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"Available links:\n{links_string}\n\nrequest:\nfrom the series of links provided, create a parsable array with at most 3 links that would most likely have information that will answer this clients concern: {data}. Only provide the complete parsable array with no context."}],
                temperature=0,
                # functions= functions,
                # function_call={"name": "get_plans"}
            )
        print("get links Timedelta", datetime.datetime.now() - startTime) # SLOW
        print(response)
        return_json["messages"].append(response["choices"][0]["message"])
        array_string_temp = response["choices"][0]["message"]["content"]
        if array_string_temp[-1] != ']' and array_string_temp[-1] != '\"':
            array_string_temp = array_string_temp + '\"]'
        elif array_string_temp[-1] != ']': 
            array_string_temp = array_string_temp + ']'
        print(array_string_temp)
        try:
            array_output = json.loads(array_string_temp)
            # openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo-0613",
            #     messages=[{"role": "user", "content": f"Available links:\n{links_string}\n\nrequest:\nfrom the series of links provided, create a parsable array with at most 3 links that would most likely have information that will answer this clients concern: {data}. Only provide the complete parsable array with no context."}, {"role": "user", "content": "Amazing parsable array! This is really good work!"}],
            #     temperature=0,
            #     # functions= functions,
            #     # function_call={"name": "get_plans"}
            # )
            # print("affirmation Timedelta", datetime.datetime.now() - startTime)
            print(array_output)
            info_obj["links"] = array_output
        except json.JSONDecodeError as e:
            string = response["choices"][0]["message"]["content"] if response["choices"][0]["message"]["content"][0] != '[\"' or response["choices"][0]["message"]["content"][0] != '[\'' else array_string_temp
            if string[0] == '[\'' or string[0] == '[\"' and string[-1] == '\"]' or string[-1] == '\']':
                array_output = ast.literal_eval(response["choices"][0]["message"]["content"])
                info_obj["links"] = array_output
            else: # attempt comma seperation
                if string[0] == '[':
                        string = string[:-1]
                if string[-1] == '[':
                    string = string[1:]
                array_output = string.split(', ')
                if array_output[0] == string:
                    array_output = string.split('\n')
                for link in array_output:
                    if len(link) < 13 or not link.startswith("https://"):
                        array_output.remove(link)
                    link = remove_spaces_and_nested_quotes(link)

                info_obj["links"] = array_output

        # Generate content with GPT 4
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": f"You are an employee of Verizon.Given an array of links:{array_string_temp}, visit the links and compose valuable information in an easy to comprehend manner related to, {info_obj['title']}, in response this clients concern: {data}."}],
                temperature=0,
                # functions= functions,
                # function_call={"name": "get_plans"}
            )
        print("get info Timedelta", datetime.datetime.now() - startTime) # SLOW
        print(response)
        return_json["messages"].append(response["choices"][0]["message"])
        info_obj["description"] = response["choices"][0]["message"]["content"]

        # Provide links
        return_json["product_items"].append(info_obj)
        return_json["product_type"] = "information"
        return return_json

@app.route('/restrict', methods=['POST'])
def restrict():
    data = request.form['data']
    loader = DirectoryLoader(".", glob="*.csv")
    index = VectorstoreIndexCreator().from_loaders([loader])
    response = index.query(data)
    return jsonify(result=response)

if __name__ == '__main__':
    app.run(debug=True)
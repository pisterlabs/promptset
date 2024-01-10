import pyodbc
from flask import Flask, request, jsonify
import json
import boto3
import io
import sys
from pdf2image.pdf2image import convert_from_path
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# openai.api_key  = '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'
# openai.api_key  = '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'
openai.api_key = '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'


app = Flask(__name__)

# Database configuration
db_server = 'localhost\SQLEXPRESS'
db_name = 'revkeep_3'
db_user = 'sa'
db_password = 'sa'


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_database_connection():
    connection_string = f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={db_server};DATABASE={db_name};UID={db_user};PWD={db_password}'
    return pyodbc.connect(connection_string)

def upload_to_textract(file_path, region_name='us-east-1'):
    # Create a Textract client with the specified region
    textract_client = boto3.client('textract', region_name=region_name)

    # Read the PDF file
    with open(file_path, 'rb') as file:
        pdf_data = file.read()

    # Upload the PDF to Textract
    response = textract_client.detect_document_text(Document={'Bytes': pdf_data})

    return response


def print_textract_output(textract_response):
    # Extract and print the text from the Textract response
    for item in textract_response['Blocks']:
        if item['BlockType'] == 'LINE':
            print(item['Text'])



@app.route('/readfile', methods=['GET'])
def read_and_clear_file():
    id=''
    try:
        with open('webroot\data.json', 'r') as file:
            file_content = file.read()
            data_dict = json.loads(file_content)
            id = data_dict.get("id")
            
        
        with open('webroot\data.json', 'w') as file:
            file.write("")
    
        try:
            print("inside boto3 try")
            connection = get_database_connection()
            cursor = connection.cursor()
            query = f"SELECT file_name FROM incoming_documents WHERE id = {id}"
            cursor.execute(query)
            result = cursor.fetchall()
            print(result)
            filename = result[0][0]
            cursor.close()
            connection.close()
            # converting pdf to jpg
            page=0
            # pdf to image converter
            images = convert_from_path('storage/incoming-documents/'+ filename ,500,poppler_path='C:/Program Files/poppler-23.07.0/Library/bin')
           
            for i in range(len(images)):
                images[i].save('storage/image/'+str(i)+'.jpg','JPEG')
                page=i

            # textract 
            textract_client = boto3.client(

                "textract",

                region_name="us-west-2",  # Replace with your desired region

                aws_access_key_id="AKIASWWCEJGAVA5D7LF2",

                aws_secret_access_key="Z567V9Z6ZHMTyHkxP5td7bn8XqaiegaQspPymbv0",

            )

            
            details=''
            try:
                print("inside textract try")
                print(page)
                for i in range(0,page):

                    with open('storage/image/'+str(i)+".jpg", "rb") as file:
                        image_bytes = file.read()
                    result = textract_client.detect_document_text(Document={"Bytes": image_bytes})

                    for block in result["Blocks"]:
                        if block["BlockType"] != "WORD":
                            continue

                        print(block["Text"], end=" ")
                        details+=' '+(block['Text'])
                print(details)
            except textract_client.exceptions.TextractException as e:
                # output error message if fails
                print(e)

            try:
                print("inside chatgpt try")
                prompt=f""" extract the information given below 
                            1. letter date
                            2. Days to respond
                            3. sender
                            4. recipient
                            5. letter subject
                            6. contact info
                            7. recipient address
                            8. Insurance provider 

                            give output in json 
                            ```{details}```
                        """
                # uncomment this for calling chatgpt API
        
                llm_response = get_completion(prompt)
                parsed_data = json.loads(llm_response)
                keys_list = list(parsed_data.keys())
                values_list = list(parsed_data.values())
                data_dict = dict(zip(keys_list, values_list))
                # Print the dictionary
                print(data_dict)

                file_path = 'webroot\\json\\output_data.json'

                # Write the dictionary to the JSON file
                with open(file_path, 'w') as json_file:
                    json.dump(data_dict, json_file)

                print("Data written to JSON file.")
                # llm_response = {
                #                     "letter_date": "07/01/2020",
                #                     "days_to_respond": "30",
                #                     "sender": "Telligen",
                #                     "recipient": "Lexington Regional Health Center",
                #                     "letter_subject": "Retrospective Request for Medical Records",
                #                     "contact_info": {
                #                         "phone": "1-855-638-7949",
                #                         "fax": "1-855-638-8017",
                #                         "address": "1776 West Lakes Parkway Intel West Des Moines, IA 50266",
                #                         "website": "telligen.nemedicalauth.com"
                #                     },
                #                     "recipient_address": "Attn: Medical Records Department 1201 N Erle St PO Box 980 . PAY Lexington, NE 68850-1571",
                #                     "insurance_provider": "Medicare"
                #                 }
                # parsed_response = json.loads(llm_response)
                print("TYPE OF RESPONSE",type(llm_response))
                with open('webroot\json\letter_data.txt', 'w') as json_file:
                    json.dump(llm_response, json_file)
                
                
                
                    print("file written")
            except Exception as e:
                print(e)

            
        except Exception as e:
            # database Exception handling
            return str(e), 500



        # for returning the data
        return data_dict
    except Exception as e:
        return str(e), 500



if __name__ == '__main__':
    app.run(debug=True)

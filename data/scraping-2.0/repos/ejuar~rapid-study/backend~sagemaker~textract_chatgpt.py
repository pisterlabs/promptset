import os
import io
import json
import re
import sys
import subprocess

import openai
import boto3

def main(file_name):
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    try:
        file_path = os.path.join(input_path, file_name)
        output_filename = re.sub(r'\.[^.]*$', '', file_name) + '.txt'
        output_file_path = os.path.join(output_path, output_filename)

        chatgpt_to_s3(
            get_secret(), 
            file_to_textract(file_path),
            output_file_path
        )
        return {"status": "Success", "message": "File processed successfully"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

def file_to_textract(file_path):
    textract = boto3.client('textract')
    with open(file_path, 'rb') as document:
        return textract.detect_document_text(
            Document={'Bytes': document.read()}
        )

def chatgpt_to_s3(api_key, textract_response, output_file_path):
    os.environ['OPENAI_API_KEY'] = api_key

    prompt = "Summarize the notes below to create an easy to understand study guide:\n" + json.dumps(textract_response)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': prompt}]
    )
    output = response['choices'][0]['message']['content'].strip()

    with io.open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(output)

def get_secret():
    session = boto3.session.Session()
    client = session.client(
        service_name = 'secretsmanager',
        region_name = "us-east-1"
    )
    return client.get_secret_value(SecretId="rapid-study")['chatgpt-key']

if __name__ == "__main__":
    file_name = sys.argv[1]  # Get file name from command line argument
    result = main(file_name)
    print(result)  # output captured by Lambda
    
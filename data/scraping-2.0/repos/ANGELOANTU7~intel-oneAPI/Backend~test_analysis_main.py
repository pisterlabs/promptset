import openai
import boto3
import json
import requests
import os

openai.api_key = ''

s3_access_key = ""
s3_secret_access_key = ""
s3_bucket_name = "learnmateai"
s3 = boto3.client(
    "s3",
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_access_key
)


def conv_pdf(json_data,filename):
    data = json.loads(json_data)
    questions = data['questions']
    topics = data['important_topics']

    #print(questions + topics)
    # Create the content for the document
    content = '<h1>Questions</h1>'
    for i, question in enumerate(questions):
        content += f'<h2>{i+1}. {question["question"]}</h2>'
        content += f'<p>Frequency: {question["frequency"]}</p>'

    content += '<h1>Important Topics</h1>'
    for topic in topics:
        content += f'<h2>{topic}</h2>'

    # Convert content to HTML
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Document</title>
    </head>
    <body>
    {content}
    </body>
    </html>
    '''
    print(filename)
    # Make a POST request to PDFShift API
    api_key = 'f894dbd8a6074a0db44439561e73c0e3'
    pdfshift_url = 'https://api.pdfshift.io/v3/convert/pdf'

    payload = {
        'source': html_content,
        'landscape': False,
        'use_print': False
    }

    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(
        pdfshift_url,
        auth=('api', api_key),
        data=json.dumps(payload),
        headers=headers
    )
    
    s3.put_object(
            Body=response.content,
            Bucket=s3_bucket_name,
            Key=f'{filename}.pdf'
        )







def read_files_from_s3_folder(bucket_name,folder_path1,folder_path2):
    response1 = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path1)
    response2 = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path2)

    if "Contents" in response1:
        for obj in response1["Contents"]:
            if obj["Key"].lower().endswith('.txt'):
                file_obj = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
                file_content = file_obj['Body'].read().decode('utf-8')
                # Process the file content here
                file_n = os.path.splitext(obj["Key"])[0]
                print(f"File: {file_n}")
                #print(file_content)
                analysis(file_n,file_content)
                #conv_pdf(file_content,file_n)
                print("-------------------")
    if "Contents" in response2:
        for obj in response2["Contents"]:
            if obj["Key"].lower().endswith('.json'):
                file_obj = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
                file_content = file_obj['Body'].read().decode('utf-8')
                # Process the file content here
                file_n = os.path.splitext(obj["Key"])[0]
                print(f"File: {file_n}")
                #print(file_content)
                # analysis(file_n,file_content)
                conv_pdf(file_content,file_n)
                print("-------------------")
                





def analysis(file_name,file_data):
    prompt = """you need to analyze a given dataset and return a table with questions appearing more than once. The table should strictly contain questions repeating more than once. The table should also contain the frequency of occurance.Also show the important topics as a diffenrent section at the bottom of the table which is of list format.make the final result in json format.The name of field showing the value of count should be frequency.All the information must be included in the result under all circumstances.It is important that the output maintains this output format under any circumstances.Only give the json file. Dont return anything other than json.
the dataset is as follows:"""

   


    # def read_object_from_s3(bucket_name, object_key):
    #     response = s3.get_object(Bucket=bucket_name, Key=object_key)
    #     object_content = response['Body'].read().decode('utf-8')
    #     return object_content

# Usage example
    bucket_name = s3_bucket_name
    #object_key = f"Sorted_PYQS/Module1.txt"
    # print(object_key)
    # object_content = read_object_from_s3(bucket_name, object_key)
    #print(object_content)

    data = file_data
    print(data)

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt + data
                    }
                ]
            )
    important_topics = response.choices[0].message.content

    print(important_topics)

    dat = important_topics

    p1 = """convert this to json and dont miss any information provided under any circumstances."""


    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": p1 + dat
                    }
                ]
            )
    res = response.choices[0].message.content
    print(res)
    resd = json.loads(res)
    print("\n\nProcessed")
    print(resd)
    file_name = file_name.split("/")[-1]
    file_path = f"{file_name}"

    # Save the data as JSON in the specified file
    with open(file_path, "w") as json_file:
        json.dump(resd, json_file)

    print(f"JSON file saved successfully at {file_path}")


    with open(f'{file_path}', 'r') as file:
        data = json.load(file)
    print(f'ModuleAnalysis/{file_name}.json')
    s3.put_object(
            Body=json.dumps(resd),
            Bucket=s3_bucket_name,
            Key=f'ModuleAnalysis/{file_name}.json'
        )
    print(data)

folder_path1 = "Sorted_PYQS/"
folder_path2 = "ModuleAnalysis/"

read_files_from_s3_folder(s3_bucket_name,folder_path1,folder_path2)
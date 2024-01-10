import openai
import requests
import json
import os
import boto3

def lambda_handler(event, context):

    def get_secret(secret_name):
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name='eu-central-1')

        try:
            response = client.get_secret_value(SecretId=secret_name)
            secret_data = json.loads(response['SecretString'])
            return secret_data
        except Exception as e:
            raise e

    def summarize_text_from_database():
        # Connect to S3
        s3 = boto3.client('s3')

        # Specify the S3 bucket and object name
        bucket_name = 'dataversionlambda'
        object_name = 'changed_data.json'

        # Retrieve the data from the S3 object
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        data = json.loads(response['Body'].read().decode('utf-8'))

        id_desc = []

        for place in data:
            if "description" in place:
                id_desc.append((place["id"], place["description"]))
            else:
                id_desc.append((place["id"], ""))
        return id_desc

    def summarize_text(text):
        prompt = "Summarize the following text in one or two sentences: \"" + text + "\" and create an interesting teaser"

        # Rufe den OpenAI API-Schlüssel aus dem Secrets Manager ab
        secret_name = "apiDatabaseKeys"
        secrets = get_secret(secret_name)

        # Zugriff auf den OpenAI API-Schlüssel
        openai_api_key = secrets['OPENAI_API_KEY']
        openai.api_key = openai_api_key

        # Generate text summary
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            stop=None,
        )

        # Extract the summarized text from the API response
        summarized_text = response.choices[0].text.strip()

        return summarized_text

    def insert_short_description(id, text):
        if text:
            url = "https://api.histourists-lhm.dpschool.app/editShortDescription/" + id

            payload = json.dumps({
                "short_description": text
            })
            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.request("PUT", url, headers=headers, data=payload)

            return response.status_code

    # Example usage
    original_texts = summarize_text_from_database()
    num_short_descriptions = 0

    for id, original_text in original_texts:
        # Generate the summarized text if the original text is not empty
        if original_text:
            # Generate the summarized text
            summarized_text = summarize_text(original_text)

            # Insert the summarized text into the short_description field for the corresponding id
            if summarized_text:
                status_code = insert_short_description(id, summarized_text)
                if status_code == 200:
                    num_short_descriptions += 1

            print("ID: ", id)
            print("Original text: ", original_text)
            print("Summarized text: ", summarized_text)
            print("-------------------------")

    return {
        "statusCode": 200,
        'body': json.dumps({
            'num_short_descriptions': num_short_descriptions
        })
    }

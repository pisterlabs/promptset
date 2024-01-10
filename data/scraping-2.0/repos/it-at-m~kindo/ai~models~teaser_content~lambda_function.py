import openai
import requests
import json
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

    def description_text_from_database():
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

    def generate_teaser(description):
        # Set up OpenAI API credentials
        # Rufe den OpenAI API-Schlüssel aus dem Secrets Manager ab
        secret_name = "apiDatabaseKeys"
        secrets = get_secret(secret_name)

        # Zugriff auf den OpenAI API-Schlüssel
        openai_api_key = secrets['OPENAI_API_KEY']
        openai.api_key = openai_api_key

        # Generate the teaser
        prompt = "Provide a creative teaser in one short statement about the place with the following description and without saying the name of the place:\n\"" + description + "\""
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=20,  # Increase the max_tokens for a longer teaser
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            stop="."  # Stop the generation at the end of a sentence
        )

        # Extract the generated teaser from the API response
        generated_teaser = response.choices[0].text.strip()

        if not generated_teaser.endswith("!"):
            generated_teaser += "!"  # Add an exclamation mark at the end

        return generated_teaser

    def insert_teaser(id, teaser):
        if not teaser:
            return

        url = "https://api.histourists-lhm.dpschool.app/editTeaser/" + id
        payload = json.dumps({
            "teaser": teaser
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.put(url, headers=headers, data=payload)
        print(response.text)

    # Fetch descriptions of places from the database (S3)
    original_texts = description_text_from_database()

    for id, original_text in original_texts:
        # Generate the teaser for the place description
        teaser = generate_teaser(original_text)

        # Insert the teaser into the place's field
        insert_teaser(id, teaser)

        print("Place ID:", id)
        print("Generated Teaser:", teaser)
        print("-------------------------")

    return {
        "statusCode": 200,
        "body": "Teaser generation and insertion completed successfully!"
    }

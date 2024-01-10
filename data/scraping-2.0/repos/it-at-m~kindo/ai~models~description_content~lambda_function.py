import openai
import requests
import json
import boto3

# Function to get the OpenAI API key from AWS Secrets Manager
def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name='eu-central-1')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_data = json.loads(response['SecretString'])
        return secret_data
    except Exception as e:
        raise e

# Function to save data to S3
def save_data_to_s3(bucket_name, file_name, data):
    s3 = boto3.client('s3')
    s3.put_object(Body=json.dumps(data), Bucket=bucket_name, Key=file_name)

# Function to update the description data in S3
def update_description_data(changed_data):
    # S3 Bucket-Name und Dateiname für die alte Daten-JSON
    bucket_name = 'dataversionlambda'
    file_name = 'changed_data.json'

    # Aktualisierte Daten in S3 speichern
    save_data_to_s3(bucket_name, file_name, changed_data)

# Function to retrieve description data from the database
def description_text_from_database():
    # Connect to S3
    s3 = boto3.client('s3')

    # Specify the S3 bucket and object name
    bucket_name = 'dataversionlambda'
    object_name = 'new_data.json'

    # Retrieve the data from the S3 object
    response = s3.get_object(Bucket=bucket_name, Key=object_name)
    new_data = json.loads(response['Body'].read().decode('utf-8'))

    loaded_data = []

    # Loop through each item in new_data
    for new_item in new_data:
        # Add the item to the loaded_data list
        loaded_data.append({
            "id": new_item["id"],
            "name": new_item["name"],
            "description": new_item["description"],
            "imageUrl": new_item["imageUrl"],
            "teaser": new_item["teaser"]
        })

    return loaded_data

# Function to generate text using OpenAI API
def generate_text(prompt):
    # Set up OpenAI API credentials
    secret_name = "apiDatabaseKeys"
    secrets = get_secret(secret_name)

    # Zugriff auf den OpenAI API-Schlüssel
    openai_api_key = secrets['OPENAI_API_KEY']
    openai.api_key = openai_api_key

    # Generate the text
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        stop=None,
    )

    # Extract the generated text from the API response
    generated_text = response.choices[0].text.strip()

    return generated_text

# Function to fetch description from the internet using OpenAI API
def fetch_description_from_internet(name):
    prompt = "Generate an eloquently and true description for the place \"" + name + "\" which is located in Munich."
    generated_description = generate_text(prompt)

    return generated_description

# Function to paraphrase the description
def paraphrase_description(name, description):
    prompt = "Generate an eloquently and true description for the place \"" + name + "\" which is located in Munich. Only use the \"" + description + "\" for that if it´s linked to \"" + name + "\" otherwise disregard it."
    generated_description = generate_text(prompt)

    return generated_description

# Function to insert the generated description into the API
def insert_description(id, generated_description):
    url = "https://api.histourists-lhm.dpschool.app/editDescription/" + str(id)
    payload = json.dumps({
        "description": generated_description
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.put(url, headers=headers, data=payload)
    print(response.text)

def lambda_handler(event, context):
    # Beispielverwendung
    loaded_data = description_text_from_database()

    for index, item in enumerate(loaded_data):
        id = item["id"]
        description = item["description"]
        name = item["name"]

        if description == "" or description is None:
            # Fetch the description from the internet
            generated_description = fetch_description_from_internet(name)

            # Insert the fetched description into the description field for the corresponding id
            insert_description(id, generated_description)

        # Check if the generated description is less than 50 characters
        if len(description) < 50:
            # Paraphrase the description
            generated_description = paraphrase_description(name, description)

            # Insert the paraphrased description into the description field for the corresponding id
            insert_description(id, generated_description)

        # Update the description field in loaded_data with the generated or paraphrased description
        loaded_data[index]["description"] = generated_description

    # Update the description data in S3
    update_description_data(loaded_data)

    return {
        "statusCode": 200,
        "body": json.dumps(loaded_data)
    }

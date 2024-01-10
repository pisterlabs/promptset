import os
import openai
import boto3
import json
from datetime import datetime
import logging
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Retrieve secrets from AWS Secrets Manager
def get_secret(secret_name):
    client = boto3.client(service_name='secretsmanager')
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        raise e
    else:
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret


def generate_recipe(ingredients, language, units):
    ingredients_list = ', '.join(ingredients)
    prompt = f"Given the following ingredients: {ingredients_list} - give me a recipe. Note that the entire recipe including the title, ingredients, and instructions must be written in {language} language. The recipe should use {units} units, and assume I have all spices. Please format the response with '1_2_3:' followed by the title, '2_3_4:' followed by the list of ingredients, '3_4_5:' followed by the instructions."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content


# Store the recipe in DynamoDB
def store_recipe(user_id, date, recipe_response, table):
    parts = recipe_response.split('2_3_4:')
    title = parts[0].replace('1_2_3:', '').strip()
    ingredients_instructions = parts[1].strip() if len(parts) > 1 else ''

    parts = ingredients_instructions.split('3_4_5:')
    ingredients = parts[0].strip()
    instructions = parts[1].strip() if len(parts) > 1 else ''

    item = {
        'user_id': user_id,
        'date': date,
        'title': title,
        'ingredients_list': ingredients,
        'instructions': instructions
    }
    table.put_item(Item=item)


def lambda_handler(event, context):
    # CORS headers
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    }

    # Handle OPTIONS requests for CORS
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers
        }

    # Set up DynamoDB
    environment = os.getenv("ENVIRONMENT")
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(environment)

    # Handling GET request to fetch past recipes for a user
    if event['httpMethod'] == 'GET':
        try:
            user_id = event['queryStringParameters']['user_id']
            # Fetch all the recipes for a specific user from DynamoDB
            response = table.query(
                KeyConditionExpression=Key('user_id').eq(user_id)
            )
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps(response['Items'])
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': str(e)})
            }

    # Handling POST request to create a new recipe
    elif event['httpMethod'] == 'POST':
        try:
            # Parse the 'body' field from the event object
            body = json.loads(event.get('body', '{}'))

            # Extract 'user_id', 'ingredients_list', 'language' and 'units' from the body
            user_id = body.get('user_id')
            ingredients = body.get('ingredients_list')
            language = body.get('language')
            units = body.get('units')

            if not user_id or not ingredients or not language or not units:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'message': 'user_id, ingredients_list, language, and units are required'})
                }

            # Dynamically build the secret_name based on the environment
            secret_name = f"{environment}_secret"
            secrets = get_secret(secret_name)

            # Set OpenAI credentials from secrets
            openai.organization = secrets['openai_org']
            openai.api_key = secrets['openai_key']

            # Generate recipe
            generated_recipe = generate_recipe(ingredients, language, units)

            # Store recipe in DynamoDB
            current_date_string = datetime.now().strftime('%Y-%m-%d')
            store_recipe(user_id, current_date_string, generated_recipe, table)

            # Return a success response with CORS headers
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'message': 'Recipe generated and stored successfully'})
            }

        except Exception as e:
            logger.error("Error processing request: {}".format(e))
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'message': 'Internal Server Error'})
            }

    else:
        return {
            'statusCode': 400,
            'headers': cors_headers,
            'body': json.dumps({'message': 'Invalid request method'})
        }

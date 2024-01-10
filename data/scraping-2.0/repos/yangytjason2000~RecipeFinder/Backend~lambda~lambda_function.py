import boto3
from boto3.dynamodb.conditions import Key
import json
import jwt
import re
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
METHODS = set(['GET', 'POST', 'DELETE'])
TABLES = set(['ingredient','recipe'])
ISO8601 = r'^([0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(.[0-9]{3})Z$'

def lambda_handler(event, context):
    method = event['httpMethod']
    if method not in METHODS:
        return serialize_invalid_response(f'Unsupported HTTP method: {method}')
    table_name = event['queryStringParameters']['database']
    if table_name not in TABLES:
        return serialize_invalid_response(f'Invalid resource name: {table_name}')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    username = get_username(event)
    if method == 'GET':
        if event['queryStringParameters']['mode']=='recommend':
            return recommend_item(table,username)
        else:
            return get_item(table,username)
    if method == 'POST':
        if event['queryStringParameters']['mode']=='consume':
            return consume_item(table, event['body'],username)
        else:
            return post_item(table, event['body'],username)
    if method == 'DELETE':
        return delete_item(table, event['body'],username) 
    
def generate_text(prompt):
    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt=prompt,
        temperature=0.7,
    )
    text = response.choices[0].text.strip()
    return text

def get_username(event):
    # Extract the token from the Authorization header
    authorization_header = event['headers'].get('Authorization')
    token = authorization_header.split()[1]

    # Decode the token to retrieve the user ID or username
    decoded_token = jwt.decode(token, algorithms=['HS256'],options={"verify_signature": False})
    username = decoded_token['cognito:username']
    return username

def check_availibility(fridge,ingredients):
    for ingredient in ingredients:
        if ingredient['name'] in fridge and ingredient['quantity']>fridge[ingredient['name']]:
            return False
        elif ingredient['name'] not in fridge:
            return False
    return True

def recommend_item(table,username,item_number=1):
    if item_number < 0:
        return serialize_invalid_response('Not a valid number')
    recipes = []
    response = table.query(KeyConditionExpression=Key('username').eq(username))
    recipes.extend(response['Items'])
    while 'LastEvaluatedKey' in response:
        response = table.query(KeyConditionExpression=Key('username').eq(username),
                            ExclusiveStartKey=response['LastEvaluatedKey'])
        recipes.extend(response['Items'])
    dynamodb = boto3.resource('dynamodb')
    fridge_table = dynamodb.Table('ingredient')
    fridge = {}
    response = fridge_table.query(KeyConditionExpression=Key('username').eq(username))
    for item in response['Items']:
        fridge[item['name']] = item['quantity']
    while 'LastEvaluatedKey' in response:
        response = fridge_table.query(KeyConditionExpression=Key('username').eq(username),
                            ExclusiveStartKey=response['LastEvaluatedKey'])
        for item in response['Items']:
            fridge[item['name']] = item['quantity']
    recommend_recipe = []
    for recipe in recipes:
        if check_availibility(fridge,recipe['ingredient']):
            recommend_recipe.append(recipe)
    if (len(recommend_recipe)>=item_number):
        recommend_recipe = sorted(recommend_recipe,key = lambda x: x['date'])[:item_number]
    return serialize_correct_response(recommend_recipe)


def get_item(table,username):
    items = []
    response = table.query(KeyConditionExpression=Key('username').eq(username))
    items.extend(response['Items'])
    while 'LastEvaluatedKey' in response:
        response = table.query(KeyConditionExpression=Key('username').eq(username),
                            ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
    return serialize_correct_response(items)

def consume_item(table, payload,username):
    valid, item = validate_payload(payload)
    if not valid:
        return serialize_invalid_response(item)

    if 'date' in item:
        date = item['date']
        match = re.fullmatch(ISO8601, date)
        if match is None:
            return serialize_invalid_response(f'Invalid date: {date}')
    ingredients = item['ingredient']
    for ingredient in ingredients:
        response = table.get_item(Key={'username': username,'name': ingredient['name']})
        if 'Item' not in response:
            return serialize_invalid_response('No such ingredient: '+ingredient['name'])
        if float(response['Item']['quantity'])<float(ingredient['quantity']):
            return serialize_invalid_response('Not enough ingredient: '+ingredient['name'])
    for ingredient in ingredients:
        if (float(response['Item']['quantity'])-float(ingredient['quantity'])).is_integer():
            response['Item']['quantity']=str(int(float(response['Item']['quantity'])-float(ingredient['quantity'])))
        else:
            response['Item']['quantity']=str(float(response['Item']['quantity'])-float(ingredient['quantity']))
        table.put_item(Item=response['Item'])
    return get_item(table,username)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def post_item(table, payload, username):
    valid, item = validate_payload(payload)
    if not valid:
        return serialize_invalid_response(item)

    if 'date' in item:
        date = item['date']
        match = re.fullmatch(ISO8601, date)
        if match is None:
            return serialize_invalid_response(f'Invalid date: {date}')
    if 'quantity' in item:
        quantity = item['quantity']
        if not is_number(quantity):
            return serialize_invalid_response('Invalid quantity')
    if 'ingredient' in item:
        ingredients = item['ingredient']
        for ingredient in ingredients:
            if not is_number(ingredient['quantity']):
                return serialize_invalid_response('Invalid quantity')
    item['name'] = item['name'].strip()
    item['username'] = username
    table.put_item(Item=item)
    return get_item(table,username)

def delete_item(table, payload,username):
    valid, item = validate_payload(payload)
    if not valid:
        return serialize_invalid_response(item)

    table.delete_item(Key={'username': username,'name': item['name']})
    return get_item(table,username)

def validate_payload(payload):
    try:
        payload = json.loads(payload)
    except Exception:
        return False, 'Payload not valid'
    if not 'name' in payload:
        return False, 'Ingredient name not provided'
    return True, payload

def serialize_correct_response(body=None):
    return {
        'statusCode': 200,
        'headers': {},
        'body': json.dumps(body),
        'isBase64Encoded': False
    }

def serialize_invalid_response(message):
    body = {
        'message': message
    }
    return {
        'statusCode': 400,
        'headers': {},
        'body': json.dumps(body),
        'isBase64Encoded': False
    }
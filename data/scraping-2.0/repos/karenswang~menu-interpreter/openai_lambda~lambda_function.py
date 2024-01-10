import json
import boto3
import openai
from openai import OpenAI
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
import os

HOST = 'search-restaurant-menus-ts66x77o3tq7lapsmpgpjjcqli.us-east-1.es.amazonaws.com'
REGION = 'us-east-1'
service = 'es'
INDEX = 'menus'

def lambda_handler(event, context):
    try:
        print("event: ", event)

        # Check if 'body' key exists in the event
        if 'body' not in event:
            return {
                'statusCode': 400,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps('No body in the request')
            }

        body = json.loads(event['body'])
        username = body['username']
        menu_id = event['pathParameters']['menu_id']

        dynamodb = boto3.resource('dynamodb')
        user_table = dynamodb.Table('user')
        # menu_table = dynamodb.Table('menu-items')
        
        openai.api_key = os.environ['OPENAI_API_KEY']
        
        client = OpenAI(
            api_key=openai.api_key,
            )
        
        user_data = user_table.get_item(Key={'username': username})
        if 'Item' not in user_data or not user_data['Item'].get('isLoggedIn'):
            return {
                'statusCode': 403,
                'headers': {
                'Access-Control-Allow-Origin': '*'
            },
                'body': json.dumps('User is not logged in or does not exist.')
            }
        
        preferences = user_data['Item']['preferences']
        
        print("menu id: ", menu_id)
        os_client = OpenSearch(hosts=[{
        'host': HOST,
        'port': 443
        }],
                        http_auth=get_awsauth(REGION, 'es'),
                        use_ssl=True,
                        verify_certs=True,
                        connection_class=RequestsHttpConnection)
        
        search_response = os_client.search(
                    index='menus',
                    body={
                        'query': {
                            'term': {
                                'restaurant_name': menu_id
                            }
                        }
                    }
                )
        print("search response: ", search_response)
        if not search_response['hits']['hits']:
            return {
                'statusCode': 404,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps('Menu does not exist.')
            }
        menu_data = search_response['hits']['hits'][0]['_source']
        menu_items = menu_data['menu_text']
        restaurant_name = menu_data['restaurant_name']
        
        # menu_data = menu_table.get_item(Key={'menu_id': menu_id})
        # if 'Item' not in menu_data:
        #     return {
        #         'statusCode': 404,
        #         'headers': {
        #         'Access-Control-Allow-Origin': '*'
        #     },
        #         'body': json.dumps('Menu does not exist.')
        #     }
            
        # menu_items = menu_data['Item']['menu_text']
        # restaurant_name = menu_data['Item']['restaurant_name']
        
        report = generate_report(client, menu_items, restaurant_name, preferences)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(report)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(f'An error occurred: {str(e)}')
        }
    
def generate_report(client, menu_items, restaurant_name, preferences):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You have information about a restaurant's dishes and a user's dietary preferences."},
        {"role": "user", "content": f"Menu items: {menu_items}\nRestaurant name: {restaurant_name}\nUser preferences: {json.dumps(preferences)}\n\nGenerate a report on the matchness of the menu to the user preferences. \
        The report should include the following sections:\n\n1. Resteraunt Name \n\n2. Favorites: Highlight dishes that match the user food preference but do not contain user dieary restrictions.\n\n3. Dishes with Allergen Warning: Identify dishes that contains users dieary restrictions.\
        Use this as an example:\
        Restaurant: David Burke Tavern\nMost dominant food category: seafood (80%)\
        Recommended: cauliflower steak, roasted rack of lamb, wild mushroom ravioli\
        Avoid: fire roasted halibut, diver scallops, lobster salad\
        Your match: low\
        Reasoning: based on your profile, you should avoid food with high uric acid. Since most dishes in David Burke Tavern are seafood, we do not recommend dining here."}
    ]
    )
    answer = completion.choices[0].message.content
    return answer

def get_awsauth(region, service):
    cred = boto3.Session().get_credentials()
    return AWS4Auth(cred.access_key,
                    cred.secret_key,
                    region,
                    service,
                    session_token=cred.token)

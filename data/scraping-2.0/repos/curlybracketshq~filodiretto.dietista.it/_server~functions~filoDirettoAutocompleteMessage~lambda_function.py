import json
from common import auth
from common import cors
from common import errors
from common import openai

print('Loading function')


@cors.access_control(methods={'POST'})
@errors.notify_discord
@auth.require_auth
def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    if event['httpMethod'] != 'POST':
        return {"statusCode": 405, "body": "Method not allowed"}

    body = json.loads(event['body'])
    result = openai.chat_completions(body['message'])
    if 'choices' in result and result['choices']:
        content = result['choices'][0]['message']['content']
    else:
        content = ''

    return {"statusCode": 200, "body": json.dumps({'content': content})}

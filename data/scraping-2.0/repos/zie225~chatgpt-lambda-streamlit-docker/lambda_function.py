import os
import json
import openai

def lambda_handler(event, context):
    prompt = event["queryStringParameters"]["prompt"]
    KEY = os.environ["API_KEY"]
    openai.api_key = KEY
    model_engine = "text-davinci-003"
    response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=500)
    text_result = response.choices[0].text
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(text_result,ensure_ascii=False,indent=4)
    }

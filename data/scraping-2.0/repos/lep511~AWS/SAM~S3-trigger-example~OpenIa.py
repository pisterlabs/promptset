import openai
import json

openai.api_key = "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"

def lambda_handler(event, context):
    # TODO implement
    print(event)
    try:
        text_ingress = json.dumps(event['text_tofunction'])
        print(text_ingress)
        result = ai_function(text_ingress)
    except:
        return {
            'statusCode': 400,
            'body': json.dumps("Not found text_tofunction parameter")
        }
    else:
        return {
            'statusCode': 200,
            'body': result
        }


def ai_function(text_function):
      
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text_function,
        temperature=0,
        max_tokens=260,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["**"]
    )

    text_out = response["choices"]
    return text_out
    

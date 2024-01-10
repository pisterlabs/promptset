import os
import openai


def lambda_handler(event, context):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        engine="davinci",  # update as per: https://stackoverflow.com/questions/65667929/invalidrequesterror-must-provide-an-engine-parameter-while-invoking-openai-ap
        prompt=event["code"],
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
    )

    return {
        'statusCode': 200,
        'body': response
    }
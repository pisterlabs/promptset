import json
import os
from openai import OpenAI
import boto3


def lambda_handler(event, context):
    try:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        s3_client = boto3.client("s3")

        response = s3_client.get_object(
            Bucket=os.environ["INSTRUCTIONS_BUCKET"], Key="instructions.txt"
        )
        instructions = response["Body"].read().decode("utf-8")

        system_prompt = [{"role": "system", "content": instructions}]

        body = json.loads(event["body"])
        conversation = body.get("conversation", [])

        response = openai_client.chat.completions.create(
            model=os.environ["MODEL_ID"], messages=system_prompt + conversation
        )
        response_message = response.choices[0].message.content

        return build_response(status_code=200, body={"response": response_message})

    except Exception as error:
        print(f"Error: {error}")
        return build_response(status_code=500, body={"error": "Internal server error"})


def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
        },
        "body": json.dumps(body),
    }

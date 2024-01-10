import json
import openai
import boto3
import time
from datetime import datetime

# Initialize AWS IoT client
def create_aws_iot_client():
    iot_client = boto3.client('iot-data', region_name='ap-southeast-2', aws_access_key_id='{ENTER_YOUR_ACCESS_KEY_HERE}', aws_secret_access_key='ENTER_YOUR_SECRET_ACCESS_KEY_HERE')  # replace 'ap-southeast-2' with your AWS region
    return iot_client

# Initialize OpenAI client
def interact_with_chatgpt(prompt):
    openai.api_key = '{ENTER_OPENAI_API_KEY_HERE}'
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )

    return response.choices[0].text.strip()

def publish_to_aws_iot_topic(iot_client, topic, message):
    json_message = json.dumps({"message": message})
    return iot_client.publish(
        topic=topic,
        qos=0,
        payload=json_message
    )

def main():
    prompt = "Tell a joke of the day"
    topic = "sensor/chat1"

    iot_client = create_aws_iot_client()

    while True:
        chatgpt_response = interact_with_chatgpt(prompt)
        publish_response = publish_to_aws_iot_topic(iot_client, topic, chatgpt_response)
        print(f"{datetime.now()}: Published message to AWS IoT topic: {topic}")
        time.sleep(300)

if __name__ == "__main__":
    main()

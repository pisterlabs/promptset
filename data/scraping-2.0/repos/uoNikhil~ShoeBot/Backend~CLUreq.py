import requests
import json
from config import api_url, subscription_key, request_id, openAiKey
from openai import OpenAI

def call_clu_model(participant_id, text, language, subscription_key, request_id, api_url):
    """
    Call the CLU model and return the response.

    :param participant_id: The participant ID for the conversation item.
    :param text: The text to be analyzed.
    :param language: The language of the query.
    :param subscription_key: The subscription key for the CLU model API.
    :param api_url: The URL of the CLU model API.
    :return: The response from the CLU model.
    """
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Apim-Request-Id': request_id,  # Replace or generate as needed
        'Content-Type': 'application/json'
    }

    data = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": participant_id,
                "text": text,
                "modality": "text",
                "language": language,
                "participantId": participant_id
            }
        },
        "parameters": {
            "projectName": "FindProductIntent",  # Replace with your project name
            "verbose": True,
            "deploymentName": "deployShoeBotv1",  # Replace with your deployment name
            "stringIndexType": "TextElement_V8"
        }
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to call CLU model: {response.status_code} - {response.text}")
        return None

def handle_clu_response(result):
    top_intent = result['result']['prediction']['topIntent']
    response = {}

    if top_intent == 'FindProductIntent':
        entities = result['result']['prediction']['entities']
        response = {entity['category']: entity['text'] for entity in entities}

    elif top_intent == 'SuggestShoes':
        query = result['result']['query']
        response = call_openai_api(query)

    elif top_intent == 'None':
        response = "I didn't understand that, please explain it differently."

    return response



def call_openai_api(text):
    client = OpenAI(api_key=openAiKey)  # Replace with your actual API key

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Replace with your preferred model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        # Assuming the last message in the response will be the assistant's reply
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


# Example usage

participant_id = "test_user"  # Set this as appropriate
text_to_analyze = "What is the difference between football studs and cleats"
language = "en"  # Replace with the appropriate language code

result = call_clu_model(participant_id, text_to_analyze, language, subscription_key, request_id, api_url)
response = handle_clu_response(result)



if result is not None:
    #print(json.dumps(result, indent=4))
    print("------------------------------")
    response = handle_clu_response(result)
    print(response)

else:
    print("Error calling the CLU model")

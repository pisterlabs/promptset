'''Script to interact with DALL-E API.'''
# %%
import boto3
import json
import openai
import os
import requests
import base64
import time


def generate_image(event, context) -> None:
    '''Score feature JSON object with 3 properties.

    - img_prompt
    - img_url
    - img_data
    '''
    message = event['Records'][0]['Sns']['Message']
    print(f'Received SNS message: {message}')

    feature_json_name = f'feature_{message}.json'

    s3 = boto3.client('s3')
    response = s3.get_object(
        Bucket='glimpse-feature-store', Key=feature_json_name
    )
    features = response['Body'].read().decode('utf-8')
    features = eval(features)

    initial_prompt = features['selected_section']

    # GPT prompt
    openai.api_key = os.environ['openai_key']
    leonardo_api_key = os.environ['leonardo_api_key']

    prompt_template = '''
    Using the following snippet encased in ```, generate an image prompt of an imaginery character  that is relevant to the snippet, be extravagent and detailed in the description of this character including the type of creature (if applicable) , race, age group, ethnicity and appearance. The character need not seem like someone that exists in the world currently.
    Also come up with a creative backstory in around 50 words on this character's origin story and his/her/its main superpower, and include some elements of the original snippet in the backstory.

    ```Bangladesh Arrest Thousands in 'Violent' Crackdown: HRW```
    '''  # noqa: E501

    response_template = '''
    Description:
    Oracle Lumineer, an ethereal being, appears as an ageless cosmic seer with radiant iridescent skin. Their eyes, gleaming with interstellar wisdom, reflect the struggles of oppressed souls. Adorned in celestial robes, they embody the resilience of hope in the face of darkness.

    Backstory:
    Originating from a cosmic realm, Oracle Lumineer descended to Earth as a response to cries for justice. Infused with the cosmic energy of empathy, they can traverse time and space, seeking out injustice to intervene and inspire change.

    Main Superpower:
    Cosmic Empathy - Oracle Lumineer possesses the ability to empathize with the collective suffering of oppressed individuals. Drawing on cosmic energies, they channel empathy to influence hearts and minds, fostering unity and inspiring resistance against systemic injustice.
    '''  # noqa: E501

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'system',
                'content': 'You are a creative character designer'
            },
            {
                'role': 'user',
                'content': prompt_template
            },
            {
                'role': 'assistant',
                'content': response_template
            },
            {
                'role': 'user',
                'content': f'Do the same for the following snippet: `{initial_prompt}`'  # noqa: E501
            }
        ]
    )

    image_prompt = completion.choices[0].message.content

    prompt = image_prompt.replace('\n', '').split(
        'Description:')[-1].split('Backstory:')[0]

    features['img_prompt'] = prompt

    # Leonardo API

    url = "https://cloud.leonardo.ai/api/rest/v1/generations"

    payload = {
        "height": 512,
        "modelId": "1e60896f-3c26-4296-8ecc-53e2afecc132",
        "prompt": prompt,
        "width": 512,
        "alchemy": True,
        "highResolution": True,
        "nsfw": True,
        "num_images": 1,
        "photoReal": False,
        "presetStyle": "CINEMATIC",
        "expandedDomain": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {leonardo_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)

    for k in range(25):
        print(f'Sleeping...{k}')
        time.sleep(1)

    gen_id = response.json()['sdGenerationJob']['generationId']

    get_gen_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}"

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {leonardo_api_key}"
    }

    response = requests.get(get_gen_url, headers=headers)

    img_url = (
        response.json()
        ['generations_by_pk']
        ['generated_images'][0]['url']
    )

    img_bytes = requests.get(img_url).content
    img_byte_str = base64.b64encode(img_bytes).decode('utf-8')

    features['img_url'] = img_url
    features['img_byte_str'] = img_byte_str

    # score JSON file with model results
    s3.put_object(
        Bucket='glimpse-feature-store',
        Key=feature_json_name,
        Body=json.dumps(features)
    )
    print('generate_image invoked')

    sns = boto3.client('sns')
    topic_arn = 'arn:aws:sns:ap-southeast-2:906384561362:glimpse-img-gen-sns'
    sns.publish(TopicArn=topic_arn, Message=message)

    print('Msg published to glimpse-img-gen-sns')

    return None

# %%

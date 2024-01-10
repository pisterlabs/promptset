import openai

openai.api_key = ""


def lambda_handler(event, context):
    q = event.get('queryStringParameters', {}).get('prompt', '')

    return_message = ''
    if q:
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=q,
            temperature=0,
            max_tokens=100,
            top_p=1,
        )
        # print(response)
        print(response.choices[0].text.strip())
        return_message = response.choices[0].text.strip()

    return {
        'statusCode': 200,
        'body': return_message
    }
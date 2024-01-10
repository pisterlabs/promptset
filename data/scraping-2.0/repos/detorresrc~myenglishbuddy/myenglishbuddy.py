from openai import OpenAI

from chatgpt import generate_short_story
from common import get_ai_message
from sender import send_email
from database import add_new_word, get_words, initialize
import json

initialize()


def lambda_handler(event, context):
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=get_ai_message(get_words()),
            stream=False
        )

        message = completion.choices[0].message.content.strip()
        message_object = json.loads(message)
        short_story = generate_short_story(client, message_object['word_of_the_day'])
        send_email(message_object, short_story)
        add_new_word(message_object['word_of_the_day'], message, short_story)

        return {
            'statusCode': 200,
            'body': json.dumps('Success')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }

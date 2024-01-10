import json
import openai
import random
import pika
import configparser
import logging

from utils import get_single_story

config = configparser.ConfigParser()
config.read('config.ini')

AMPQ_URL = config['RabbitMQ']['AMQP_URL']

def get_response(instructions):
    messages = [
        { "role": "system", "content": instructions },
    ]

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    return completion.choices[0].message.content

def read_words_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

def generate_story_prompt(english_story):
    # Create the prompt
    prompt = f"Please translate the following English passage into Bengali. Ensure that the translation is accurate and retains the original meaning and tone of the passage. The passage reads: {english_story}"
    return prompt

def write_to_rabbitmq(story, amqp_url):
    try:
        # Parse the AMQP URL with credentials
        parameters = pika.URLParameters(amqp_url)

        # Establish a connection
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare the queue
        channel.queue_declare(queue='stories', durable=False)

        # Ensure the story is a byte string
        if isinstance(story, str):
            story = story.encode()

        # Publish the message
        channel.basic_publish(exchange='', routing_key='stories', body=story)
        logging.info("Message sent to RabbitMQ")
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"Failed to connect to RabbitMQ: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Close the connection if it's open
        if 'connection' in locals() and connection.is_open:
            connection.close()
            logging.info("RabbitMQ connection closed")

start = 0
end = 1000
i = 0
# read story from english_stories.txt each story is separated by a newline
for english_text in read_words_from_file('english_stories.txt'):
    #story_prompt = generate_story_prompt(story)
    #print(story_prompt)
    #story = get_response(story_prompt)
    #write_to_rabbitmq(story, AMPQ_URL)
    i+=1
    bangla_text = get_single_story()
    with open('test.jsonl', 'a', encoding='utf-8') as file:
        json.dump({"instruction": "Please translate the following English passage into Bengali. Ensure that the translation is accurate and retains the original meaning and tone of the passage.", "input": english_text, "output": bangla_text}, file)
        file.write('\n')
    if i == end:
        break
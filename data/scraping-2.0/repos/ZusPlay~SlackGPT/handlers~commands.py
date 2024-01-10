import re
import logging
from openai import InvalidRequestError
from slack_sdk.errors import SlackApiError
from loader import app, gpt_chain
from utils.converters import convert_slack_id_to_username, convert_output_to_message


# @app.command('/id')
# def change_settings(ack, respond, command):
#     ack()
#     respond(f'Your ID is `{command['user_id']}`.')


@app.message('ping')
def ping_pong_handler(_, say):
    say('pong')


@app.event('app_mention')
def handle_mention(event, say):
    ts = event.get('thread_ts', None) or event.get('ts')
    text = event.get('text')
    username = convert_slack_id_to_username(event.get('user'))
    get_bot_name = convert_slack_id_to_username(app.client.auth_test()['user_id'])
    tag_pattern = fr'^{get_bot_name}$'
    contact_pattern = fr'^{get_bot_name} '
    if re.match(tag_pattern, text):
        try:
            channel_id = event.get('channel')
            thread_response = app.client.conversations_replies(channel=channel_id, ts=event.get('thread_ts'))
            thread_message = thread_response.get('messages')[0].get('text')
            output = convert_output_to_message(gpt_chain.predict(human_input=thread_message), username)
            say(output, thread_ts=ts)
        except SlackApiError as error:
            logging.error(error)
    elif re.match(contact_pattern, text):
        try:
            text = re.sub(contact_pattern, '', text)
            output = convert_output_to_message(gpt_chain.predict(human_input=text), username)
            say(output, thread_ts=ts)
        except InvalidRequestError as error:
            logging.critical(error)


@app.event('message')
def message_handler(message, say):
    try:
        output = gpt_chain.predict(human_input=message['text'])
        say(output)
    except InvalidRequestError as error:
        logging.critical(error)

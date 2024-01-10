import typing
import requests

import openai

from flask import current_app

from app.models import Message, Conversation


class ModelResponseError(Exception):
    """
    Raised if JWT can not be decoded or signature is expired.
    """


def get_model_response(messages: typing.List[Message]) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[message.to_dict() for message in messages],
        temperature=0.7,
    )
    model_response = response['choices'][0]['message']['content']
    return model_response


def get_latest_message(conversation: Conversation) -> typing.Optional[Message]:
    if not conversation.messages:
        return
    latest_message = Message.query.filter_by(
        conversation_id=conversation.id
    ).order_by(Message.id.desc()).first()

    return latest_message


def send_whatsapp_message(recipient_number: str, message: str) -> None:
    url = current_app.config['META_API_URL'] + current_app.config['META_ADMIN_ID'] + '/messages'
    access_token = current_app.config['META_ACCESS_TOKEN']
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "text",
        "text": {
            "preview_url": True,
            "body": message
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.text

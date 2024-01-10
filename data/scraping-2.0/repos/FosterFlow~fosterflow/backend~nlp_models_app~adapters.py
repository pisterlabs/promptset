import os
import environ

import openai
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from project.settings import BASE_DIR
from chat_app.models import Message

env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))
openai.api_key = env('OPENAI_API_KEY')


class RequestHandler:
    def handle_request(self, request):
        sent_message = Message.objects.get(id=request['message_id'])
        nlp_model = sent_message.addressee_id.user_id
        if nlp_model.username == 'GPT-3.5-turbo-4k' and nlp_model.is_active:
            adapter = GptAdapter(Gpt35Turbo4KInterface, ResponseWebsocketInterface)
            adapter.generate_response(sent_message)
        elif nlp_model.username == 'GPT-3.5-turbo-16k' and nlp_model.is_active:
            adapter = GptAdapter(Gpt35Turbo16KInterface, ResponseWebsocketInterface)
            adapter.generate_response(sent_message)
        elif nlp_model.username == 'GPT-4-8k' and nlp_model.is_active:
            adapter = GptAdapter(Gpt48KInterface, ResponseWebsocketInterface)
            adapter.generate_response(sent_message)
        elif nlp_model.username == 'GPT-4-32k' and nlp_model.is_active:
            adapter = GptAdapter(Gpt432KInterface, ResponseWebsocketInterface)
            adapter.generate_response(sent_message)


class ResponseWebsocketInterface:
    def __init__(self):
        self.channel_layer = get_channel_layer()

    def send_chunk(self, chunk, group):
        async_to_sync(self.channel_layer.group_send)(group, {"type": "chat.message", "text": chunk})


class Adapter:
    def __init__(self, from_interface, to_interface):
        self.from_interface = from_interface
        self.to_interface = to_interface

    def get_messages(self, sent_message):
        previous_messages = Message.objects.filter(chat_id=sent_message.chat_id).order_by('id')[:10]

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for message in previous_messages:
            if message.owner_id == sent_message.owner_id:
                mes = {"role": "user", "content": message.message_text}
            else:
                mes = {"role": "assistant", "content": message.message_text}
            messages.append(mes)

        return messages

    def create_nlp_message(self, chat_id, owner_id, request_id):
        return Message.objects.create(
            chat_id=chat_id,
            message_text='',
            owner_id=owner_id,
            request_id=request_id,
        )


class GptAdapter(Adapter):
    def generate_response(self, sent_message):
        messages = self.get_messages(sent_message)
        generator = self.from_interface().create_generator(messages)
        nlp_message = self.create_nlp_message(sent_message.chat_id, sent_message.addressee_id, sent_message)

        complete = ""
        data = {
            "type": "chat_message_chunk",
            "chat_id": nlp_message.chat_id.id,
            "created_at": str(nlp_message.created_at),
            "request_id": sent_message.id,
            "id": nlp_message.id,
            "message_chunk": '',
            "owner_id": nlp_message.owner_id.id,
            "status": "start"
        }
        self.to_interface().send_chunk(data, str(sent_message.owner_id.id))

        for event in generator:
            try:
                if event['choices'][0]['finish_reason'] != 'stop':
                    event_text = event['choices'][0]['delta']['content']
                    complete += event_text
                    data['message_chunk'] = event_text
                    data['status'] = 'progress'

                    self.to_interface().send_chunk(data, str(sent_message.owner_id.id))
                else:
                    data['status'] = 'done'
                    data['message_chunk'] = ''
                    self.to_interface().send_chunk(data, str(sent_message.owner_id.id))
            except Exception as e:
                data['errors'] = {
                    "details": [
                        str(e)
                    ]
                }
                data['status'] = 'failed'
                self.to_interface().send_chunk(data, str(sent_message.owner_id.id))
        nlp_message.message_text = complete
        nlp_message.save()


class Gpt35Turbo4KInterface:
    def create_generator(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        return response


class Gpt35Turbo16KInterface:
    def create_generator(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            stream=True
        )
        return response


class Gpt48KInterface:
    def create_generator(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=True
        )
        return response


class Gpt432KInterface:
    def create_generator(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-4-32k",
            messages=messages,
            stream=True
        )
        return response

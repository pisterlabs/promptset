import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from django.core.cache import cache
import openai
import os

openai.api_key = os.environ['OPENAI_API_KEY']

class VideoChatConsumer(WebsocketConsumer):
  def connect(self):
    self.accept()

  def receive(self, text_data):
    data = json.loads(text_data)
    if data['action'] == 'join_room':
      self.room_group_name = data['room_name']
      print(data['room_name'], self.room_group_name)
      quantity = cache.get(f'{self.room_group_name}-quantity')
      if quantity == None:
        cache.set(f'{self.room_group_name}-quantity', 1)
        async_to_sync(self.channel_layer.group_add)(
          self.room_group_name,
          self.channel_name
        )
        self.send(json.dumps({
            'role': 'offerer',
            'room': self.room_group_name,
        }))
      elif quantity == 1:
        cache.set(f'{self.room_group_name}-quantity', 2)
        async_to_sync(self.channel_layer.group_add)(
          self.room_group_name,
          self.channel_name
        )
        self.send(json.dumps({
            'role': 'answerer',
            'room': self.room_group_name,
        }))
      elif quantity >= 2:
        self.room_group_name = None
        self.send(json.dumps({
          'role': 'full',
          'room': None
        }))
    elif data['action'] == 'store_offerer_SDP' and self.room_group_name == data['for_group']:
      cache.set(f'{self.room_group_name}-offerer_SDP', data['offerer_SDP'])
    elif data['action'] == 'send_answerer_SDP' and self.room_group_name == data['for_group']:
      async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type': 'send_answer',
                'answerer_SDP': data['answerer_SDP']
            }
          )
    elif data['action'] == 'get_offerer_SDP' and self.room_group_name == data['for_group']:
      offerer_SDP = cache.get(f'{self.room_group_name}-offerer_SDP')
      self.send(json.dumps({
          'offerer_SDP': offerer_SDP
      }))

    elif data['action'] == 'GPT_help' and self.room_group_name == data['for_group']:
      system_message = 'You are a helpful assistant.'
      messages = [
            {'role': 'system', 'content': system_message}
          ]
      for message in data['prompts']:
        if message['user'] == 'user-1':
          role = 'user'
          content = f'User 1: {message["content"]}'
        elif message['user'] == 'user-2':
          role = 'user'
          content = f'User 2: {message["content"]}'
        elif message['user'] == 'assistant':
          role = 'assistant'
          content = message['content']
        messages.append({'role': role, 'content': content})

      try:
        response = openai.ChatCompletion.create(
              model='gpt-3.5-turbo',
              messages=messages
            )
        gpt_message = response['choices'][0]['message']['content']
        gpt_message = gpt_message.replace('\n', '<br>')
      except:
        gpt_message = 'Sorry, GPT could not generate a response at the moment. Please try again.'
      async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type': 'send_gpt_message',
                'gpt_message': gpt_message
            }
          )
    elif data['action'] == 'DALLE' and self.room_group_name == data['for_group']:
      try:
        response = openai.Image.create(
              prompt=data['prompt'],
              n=1,
              size='256x256'
            )
        img_url = response['data'][0]['url']
        img_tag = f'<img src="{img_url}" alt="Generated Image">'
      except:
        img_tag = 'Sorry, this image could not be generated. Please try again.'
      async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type': 'send_gpt_message',
                'gpt_message': img_tag
            }
          )
  def disconnect(self, close_code):
    async_to_sync(self.channel_layer.group_discard)(
      self.room_group_name,
      self.channel_name
    )
    quantity = cache.get(f'{self.room_group_name}-quantity')
    if quantity == 2:
      cache.set(f'{self.room_group_name}-quantity', 1)
    elif quantity == 1:
      cache.delete(f'{self.room_group_name}-quantity')

  def send_answer(self, event):
    self.send(json.dumps({
        'answerer_SDP': event['answerer_SDP']
    }))

  def send_gpt_message(self, event):
    self.send(json.dumps({
        'gpt_message': event['gpt_message']
    }))

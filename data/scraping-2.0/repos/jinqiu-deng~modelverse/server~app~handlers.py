import tornado.web
from tornado.web import RequestHandler
import asyncio
import json
import logging
import openai
import os
from .utils import CustomOpenAIClient

class MainHandler(tornado.web.RequestHandler):
    def initialize(self, config, key_lock, key_state):
        self.config = config
        self.key_lock = key_lock
        self.key_state = key_state

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    def get(self):
        logging.info('Received GET request from %s', self.request.remote_ip)
        self.render(os.path.join(os.path.dirname(__file__), '..', 'templates', 'index.html'))


    async def decrement_key_state(self, group_name, selected_key_index):
        async with self.key_lock:
            self.key_state[group_name][selected_key_index] -= 1

    async def post(self):
        request_body_json = json.loads(self.request.body.decode('utf-8'))

        # Set the group_name to 'default_group' if it's not provided in the request
        group_name = request_body_json.get('group_name', 'default_group')

        for group in self.config.settings['groups']:
            if group['name'] == group_name:
                selected_group = group
                break

        if not selected_group:
            self.set_status(400)
            self.write({"error": "Invalid group_name provided"})
            return

        async with self.key_lock:
            logging.info('Current key_state: %s', self.key_state)
            selected_key_index = min(self.key_state[group_name], key=self.key_state[group_name].get)
            self.key_state[group_name][selected_key_index] += 1

            api_key = selected_group['keys'][selected_key_index]['key']
            custom_openai_client = CustomOpenAIClient(api_key)

        logging.info('Sending question "%s" from %s using key %s in group %s',
                     request_body_json,
                     self.request.remote_ip,
                     selected_key_index,
                     group_name)

        try:
            request_body_json = json.loads(self.request.body.decode('utf-8'))

            allowed_properties = {
                'model', 'messages', 'temperature', 'top_p', 'n', 'max_tokens',
                'presence_penalty', 'frequency_penalty', 'user', 'logit_bias',
                'stream'
            }

            filtered_request_body_json = {k: v for k, v in request_body_json.items() if k in allowed_properties}

            # Check if the stream is set to True
            stream = filtered_request_body_json.get('stream', False)

            if stream:
                # Use filtered_request_body_json for further processing with streaming
                async for message in custom_openai_client.create_chat_completion_stream(filtered_request_body_json):
                    chunk = json.dumps(message)
                    self.write(chunk)
                    await self.flush()
            else:
                # Use filtered_request_body_json for further processing without streaming
                completion = await custom_openai_client.create_chat_completion(filtered_request_body_json)
                answer = completion['choices'][0]['message']['content']

                logging.info('Generated completion "%s" for question "%s" from %s using key %s in group %s',
                             completion,
                             request_body_json,
                             self.request.remote_ip,
                             selected_key_index,
                             group_name)

                response = {
                    'completion': completion
                }

                self.set_header('Content-Type', 'application/json')
                self.write(json.dumps(response))

        finally:
            # 在请求处理完成后调用decrement_key_state方法
            await self.decrement_key_state(group_name, selected_key_index)

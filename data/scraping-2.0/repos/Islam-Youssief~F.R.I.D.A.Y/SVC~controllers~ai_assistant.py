import logging

import openai


class AssistantController:

    def execute(self, command):
        try:
            return self._do_execute(command)
        except Exception as e:
            logging.error(f'Failure when getting response from OpenAI, {str(e)}')
            return {'result': 'Sorry, I am not able to answer your question 🥹. Please check your api key in the config.json file 🤔'}

    def _do_execute(self, command, MaxToken=3000, outputs=1):
        response = openai.Completion.create(model="text-davinci-003", prompt=command, max_tokens=MaxToken, n=outputs)
        return {"result": [choice['text'].strip() for choice in response['choices']][0]}

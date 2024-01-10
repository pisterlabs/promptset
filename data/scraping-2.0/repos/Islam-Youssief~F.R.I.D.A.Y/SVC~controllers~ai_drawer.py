import logging

import openai


class DrawerController:

    def draw(self, description):
        try:
            return {'result': self._generate(description)}
        except Exception as e:
            logging.error(f'Failure when getting response from OpenAI, {str(e)}')
            return {'result': 'Sorry, I am not able to generate the image 🥹. Please check your api key in the config.json file 🤔'}

    def _generate(self, description):
        res = openai.Image.create(prompt=description, n=1, size="256x256") 
        return res["data"][0]["url"]

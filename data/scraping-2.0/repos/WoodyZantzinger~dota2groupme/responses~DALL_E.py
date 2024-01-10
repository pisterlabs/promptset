# -*- coding: utf-8 -*

from .CooldownResponse import *
import openai


class DallE(ResponseCooldown):

    message = "#dalle"

    RESPONSE_KEY = "#dalle"

    COOLDOWN = 1 * 60 * 60 * 8

    def __init__(self, msg):
        super(DallE, self).__init__(msg, self, DallE.COOLDOWN)

    def _respond(self):

        openai.api_key = DataAccess.get_secrets()['OPENAI_KEY']
        response = openai.Image.create(
            prompt=self.msg.text.partition(' ')[2],
            n=1,
            size="512x512"
        )

        return output_message.OutputMessage(response['data'][0]['url'], output_message.Services.PHOTO_URL)
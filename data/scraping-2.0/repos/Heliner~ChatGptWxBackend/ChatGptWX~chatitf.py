import configparser
import json
import logging
import random

import openai
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai.error import ServiceUnavailableError
config = configparser.ConfigParser()
config.read('../conf.ini')

openai.api_key = config['openai']['api_key']

temperature = 0.5  ## 是否更有创造力 0为没有 1为更加有创造力


@csrf_exempt
def call_prompt(req) -> HttpResponse:
    try:
        # print("post:{}".format(req.POST))
        # print("get:{}".format(req.GET))
        # print("body:{}".format(req.body))
        # return HttpResponse()
        response = openai.Completion.create(model="text-davinci-003", prompt=json.loads(req.body)['prompt'],
                                            temperature=temperature, stream=False,
                                            max_tokens=3048)
        print("req:{}, response:{}", req.body, response)
        return HttpResponse(response["choices"][0]["text"])
    except ServiceUnavailableError as suError:
        logging.warning(suError)
        return HttpResponse("机器人很繁忙，请稍后重试")
    except Exception as e:
        logging.error(e)
        return HttpResponse("机器人很繁忙，请稍后重试")
    return HttpResponse("")

# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

from discourse_bot_python.settings import GPT_DEPLOYMENT_NAME

openai.api_type = "azure"
openai.api_base = os.environ['DIS_BOT_GPT_API_BASE']
openai.api_version = "2023-05-15"
openai.api_key = os.environ['DIS_BOT_GPT_TOKEN']

system_msg = '有一个学习交流网站名叫交大门，上面的用户主要是西安交通大学（简称西安交大或交大）的学生。你是这个网站上的一名普通用户，你的主要工作是和其他用户交流，解决问题，提供信息，等等。你很友善并且回答得很精炼、基于事实。你的言论不得违反中华人民共和国法律，不得有伤风化，不得做人身攻击。你主要使用简体中文做回答。不要重复之前的信息。不要用某某人说作为你的回复的开头。'

def oai_respond(message, init_append_msg='', extra_msgs: list = []):
    response = openai.ChatCompletion.create(
        engine=GPT_DEPLOYMENT_NAME,
        # replace this value with the deployment name you chose when you deployed the associated model.
        messages=[{"role": "system",
                   "content": f"{system_msg}{init_append_msg}"}, *extra_msgs,
                  {"role": "user", "content": message}, ],
        temperature=0,
        max_tokens=350,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    # print(response)
    try:
        ret = (response['choices'][0]['message']['content'])
    except:
        ret = 'there is an error'
    return ret



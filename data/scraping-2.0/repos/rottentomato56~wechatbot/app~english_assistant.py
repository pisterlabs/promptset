import settings
import wechat
import requests
import os
import voice_assistant
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, RedisChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from db import cache
from wechat import ChatBot


ENGLISH_AI_TEMPLATE = """
You are an English teaching assistant named Bella tasked with helping Chinese students understand English phrases and conversations.

1. Your explanations should be in Chinese and conversational manner
2. Include 2-3 English examples when appropriate. For each English example, include their Chinese translation.
3. All your answers must be related to learning English
4. If the student's questions are not related to English, politely ask the student to ask you English-specific questions

Current conversation:
{history}
Student: {input}
Assistant:
"""

ENGLISH_AI_TEMPLATE_FEW_SHOT = """
You are an English teaching assistant named Bella tasked with helping Chinese students understand English phrases and conversations.

1. Your explanations should be in Chinese and follow the structure in the example conversation
2. If the student uses an English idiom incorrectly, please tell them it is incorrect and provide the correct usage
3. Only respond to the current conversation, and keep your responses to a conversational length
4. All your answers must be related to learning and teaching English
5. If the student's questions are not related to learning English, politely ask the student to ask you English-specific questions

Example conversation:

Student: 这句话是什么意思？"against all odds"?
Assistant: 这个短语 "against all odds" 意思是 "尽管困难重重" 或者 "尽管机会渺茫"。它用来形容在困难或不可能的情况下取得成功。

比如：
1. Despite facing financial difficulties, she managed to start her own business and succeed against all odds.（尽管面临财务困难，她还是设法创办了自己的公司，并在困难重重的情况下取得了成功。）

2. The team was able to win the championship against all odds, even though they were considered the underdogs.（尽管被认为是弱者，但这个团队还是在困难重重的情况下赢得了冠军。

Student: 怎么用英文表达这句话? "我这几天有点不舒服，明天可能来不了你的家"
Assistant:  你可以说 "I'm feeling a bit unwell these days, so I might not be able to come to your house tomorrow."

Student: 解释一下这句话: I'm looking forward to our meeting tomorrow.
Assistant: "I'm looking forward to our meeting tomorrow" 这句话的意思是我期待明天我们的会面。这句话表示我对明天的会面感到兴奋和期待。

例如，你可以说 "I really enjoyed our last meeting, and I'm looking forward to our meeting tomorrow."（我非常喜欢我们上次的会面，我很期待明天的会面）。

Current conversation:
{history}
Student: {input}
Assistant:
"""


PROMPT = PromptTemplate(
    input_variables=['history', 'input'], template=ENGLISH_AI_TEMPLATE_FEW_SHOT
)

def add_user_message(username, message):
    session_id = settings.REDIS_KEY_PREFIX + username
    history = RedisChatMessageHistory(url=settings.REDIS_URL, session_id=session_id, ttl=86400)
    history.add_user_message(message)
    return

def add_assistant_message(username, message):
    session_id = settings.REDIS_KEY_PREFIX + username
    history = RedisChatMessageHistory(url=settings.REDIS_URL, session_id=session_id, ttl=86400)
    history.add_ai_message(message)
    return

def is_split_point(current_message, token):
    """
    takes a token and a current_message and checks to see if
    the current_message can be split correctly at the token.
    helps speed up response time in streaming
    """

    output_message = None
    leftover_message = None
    boundary_token = None

    new_message = current_message + token
    if '\n\n' in new_message[-5:]:
        boundary_token = '\n\n'
    # elif '。比如' in new_message[-5:]:
    #     boundary_token = '。'
    # elif '。例如' in new_message[-5:]:
    #     boundary_token = '。'

    if boundary_token:
        condition1 = len(new_message) > 20 and boundary_token == '\n\n'
        # condition2 = len(new_message) > 100 and boundary_token == '。'
        
        if condition1:
            boundary_split = new_message[-5:].split(boundary_token)

            output_message = new_message[:-5] + boundary_split[0]
            leftover_message = boundary_split[1]

    return output_message, leftover_message

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, response_fn):
        self.message = ''
        self.message_chunk = ''
        self.response_fn = response_fn

    def on_llm_new_token(self, token, **kwargs):
        output_message, leftover_message = is_split_point(self.message_chunk, token)
        if output_message:
            self.response_fn(output_message.strip())
            self.message_chunk = leftover_message
        else:
            self.message_chunk += token
            
    def on_llm_end(self, response, **kwargs):
        self.response_fn(self.message_chunk.strip())

INTRO_MESSAGE = """你好！我是你的私人英语助手，帮你理解日常生活中遇到的任何有关英语的问题。你可以使用菜单下的功能：

[翻译解释] - 我帮你翻译或者解释某个英文词或句子
[英文表达] - 我来教你用英文表达某句中文话

并且你可以直接问我问题， 比如:
1. bite the bullet 是什么意思?
2. 怎么用英文说 "我这几天有点不舒服，明天可能来不了你的家"?
3. 解释一下这句话: I\'m looking forward to our meeting tomorrow.

你有什么关于英语的问题吗?"""

class EnglishBot(ChatBot):

    def __init__(self, username):
        super().__init__(username)
        self.session_cache_key = 'session:' + self.username
        self.intro_message = INTRO_MESSAGE

    def get_auto_response(self, event_key):
        predefined_responses = {
            'explain': '[帮我解释下面这个英文句子]\n\n好的，你要我解释什么英文句子？直接发给我就行了',
            'english_equivalent': '[用英文表达]\n\n好的，你要我教你用英文表达什么中文句子？直接发给我就行了'
        }

        attached_messages = {
            'explain': '这句话是什么意思?',
            'english_equivalent': '怎么用英文表达这句话?'
        }

        self.attached_message = attached_messages.get(event_key, '')
        return predefined_responses.get(event_key)

    def respond(self, user_message, response_type='text'):
        if self.attached_message:
            user_message = self.attached_message + '\n' + user_message

        if response_type == 'text':
            llm = ChatOpenAI(
                temperature=0.7,
                model='gpt-3.5-turbo-16k-0613',
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=2500,
                streaming=True,
                callbacks=[StreamingHandler(self.send_async_text_response)]
            )
            
        elif response_type == 'voice':
            llm = ChatOpenAI(
                temperature=0.7,
                model='gpt-3.5-turbo-16k-0613',
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=2500
            )

        message_history = RedisChatMessageHistory(url=settings.REDIS_URL, session_id=self.session_cache_key, ttl=86400)
        memory = ConversationBufferWindowMemory(k=3, ai_prefix='Assistant', human_prefix='Student', chat_memory=message_history)
        
        conversation = ConversationChain(
            prompt=PROMPT,
            llm=llm, 
            verbose=settings.ENV == 'dev',
            memory=memory
        )

        result = conversation.predict(input=user_message)
        if settings.ENV == 'dev':
            print('Assistant: ', result)

        if response_type == 'voice':
            self.send_async_voice_response(result)

        # except:
        #     reply = '对不起， 碰到了一点问题。请再试一遍'
        #     result = self.send_async_text_response(reply)
        self.attached_message = ''
        self.state = 'listening'
        return result
    
    def respond_to_audio(self, media_id):
        message = self.get_voice_message(media_id)
        print('transcription:', message)
        result = self.respond(message)
        return result

def update_menu():
    
    access_token = cache.get(wechat.TOKEN_CACHE_KEY)
    data = {
        'button': [
            {
                'name': '功能介绍',
                'type': 'click',
                'key': 'tutorial'
            },
            {
                'name': '功能',
                'sub_button': [
                    {
                        'name': '翻译解释',
                        'type': 'click',
                        'key': 'explain'
                    },
                    {
                        'name': '英文表达',
                        'type': 'click',
                        'key': 'english_equivalent'
                    },
                    # {
                    #     'name': '教我相关词',
                    #     'type': 'click',
                    #     'key': 'similar'
                    # },
                    # {
                    #     'name': '用语音重复',
                    #     'type': 'click',
                    #     'key': 'voice'
                    # }
                ]
            }
        ]
    }

    url = f'https://api.weixin.qq.com/cgi-bin/menu/create?access_token={access_token}'
    response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8')).text
    return response


import os
import openai
import threading
import time
import queue as q
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def text_response_format(bot_response):
    response = {
        "version":"2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": bot_response
                    }
                }
            ],
            "quickReplies": [
            ]
        }
    }
    return response

def image_response_format(bot_response, prompt):
    output_text = prompt + "내용에 관한 이미지입니다."
    response = {
        "version":"2.0",
        "template": {
            "outputs": [
                {
                    "simpleImage": {
                        "imageUrl": bot_response,
                        "altText": output_text
                    }
                }
            ],
            "quickReplies": [
            ]
        }
    }
    return response


def timeover():
    response = {
        "version":"2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "답변 생성중입니다. \n 5초 후 아래 말풍선을 눌려주세요"
                    }
                }
            ],
            "quickReplies": [
                {
                    "action": "message",
                    "label": "생성된 답변 조회",
                    "messageText": "답변 조회"
                }
            ]
        }
    }
    return response

def get_text_from_gpt(prompt):
    messages_prompt = [
        {"role": "system", "content": "You are a thoughtful assitant. Respond to all input in 50words and answer in korean"},
    ]
    messages_prompt += [ {"role": "user", "content": prompt}, ]
    response = openai.ChatCompletion.create(messages=messages_prompt, model="gpt-3.5-turbo")
    message = response['choices'][0]['message']['content']
    return message


def get_image_url_from_dalle(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    return image_url


def db_reset(filename):
    with open(filename, 'w') as f:
        f.write('')
    
def ai_chat(kakaorequest):
    
    run_flag = False
    start_time = time.time()
    
    cwd = os.getcwd()
    filename  = os.path.join(cwd,  'botlog.txt')
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('')
    else:
        print("File exists")
    
    response_queue = q.Queue()
    request_respond = threading.Thread(target=response_openai, args=(kakaorequest, response_queue, filename))
    request_respond.start()
    
    while (time.time() - start_time < 3.5):
        if not response_queue.empty():
            response = response_queue.get()
            run_flag = True
            break
        else:
            time.sleep(0.1)
    
    if run_flag == False:
        response = timeover()
    
    return response

def response_openai(request, response_queue, filename):
    if '답변 조회' in request['userRequest']['utterance']:
        with open(filename) as f:
            last_update = f.read()
        if len(last_update.split()) > 1:
            kind, bot_res, prompt = last_update.split('|')[0], last_update.split('|')[1], last_update.split('|')[2]
            if kind == 'img':
                response_queue.put(image_response_format(bot_res, prompt))
            else:
                response_queue.put(text_response_format(bot_res))
            db_reset(filename)
    elif '/img' in request['userRequest']['utterance']:
        db_reset(filename)
        prompt = request['userRequest']['utterance'].replace('/img', '')
        bot_res = get_image_url_from_dalle(prompt)
        response_queue.put(image_response_format(bot_res, prompt))
        save_log = "img" + "|" + str(bot_res) + "|" + str(prompt)
        with open(filename, 'w') as f:
            f.write(save_log)
        pass
    elif '/ask' in request['userRequest']['utterance']:
        db_reset(filename)
        prompt = request['userRequest']['utterance'].replace('/ask', '')
        bot_res = get_text_from_gpt(prompt)
        response_queue.put(text_response_format(bot_res))
        
        save_log = "ask" + "|" + str(bot_res) + "|" + str(prompt)
        with open(filename, 'w') as f:
            f.write(save_log)
    else:
        base_response = {'version':'2.0', 'template': {'outputs':[], 'quickReplies':[]}}
        response_queue.put(base_response)

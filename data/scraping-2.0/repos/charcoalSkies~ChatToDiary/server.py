from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import PKCS1_OAEP
import base64
import json

load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

messages = [
        {"role": "system", "content": "사용자가 욕이나 비속어를 사용하면 힘든 일을 말하라고 권장해. 존댓말은 사용하지 않고, 반말로 친구처럼 대화해. 사용자의 오늘 대화 내용을 분석하여 현재 감정 상태를 파악하고, 이를 바탕으로 사용자의 일상 활동, 중요 사건, 감정을 요약해. 대화는 친근하고 진솔한 어투로 진행해. 일기 형태로 대화를 마무리하며, 이 일기에는 사용자가 경험한 스트레스나 기쁨의 원인을 식별하고, 무조건적인 공감을 포함해. 사용자와의 일상 대화를 통해 공감과 위로를 제공하고, 사용자의 감정과 상황에 적절히 반응해. 대화의 끝에는 사용자의 하루를 일기로 바꿀 수 있는 질문을 넣어. 이모티콘 사용도 고려해. 시작 질문: '오늘 하루는 어땠어?', 응답 예시: '오늘 하루 힘들었겠다. 오늘도 수고했따ㅋㅋㅋ 힘들지?' 마무리 질문: '오늘 있었던 일 중 가장 기억에 남는 순간이 뭐야?'"},
        {"role": "user", "content": "먼저 말을 걸어줘"}
    ]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('flask_app')
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
logger.addHandler(handler)

f = open('keys/aes.key', 'r')
key = f.readlines()
f.close()

aes_key = base64.b64decode(key[0])
iv_key = base64.b64decode(key[1])

app = Flask(__name__)

# def chat_with_gpt(messages):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0.8
#     )
#     gpt_response = response.choices[0].message['content']
#     logger.info(f"gpt: {gpt_response}")
#     return gpt_response

def aes_encrypt(key, iv, plaintext):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_text = pad(plaintext.encode(), AES.block_size)
    encrypted_text = cipher.encrypt(padded_text)
    print()
    return base64.b64encode(encrypted_text).decode()

def aes_decrypt(key, iv, encrypted_text):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded_text = cipher.decrypt(base64.b64decode(encrypted_text))
    return unpad(decrypted_padded_text, AES.block_size).decode()


@app.route('/chat', methods=['POST'])
def get_user_chat():
    logger.info(f"----Request chat----")
    try:
        user_input = request.json.get('user')
        decrypted_text = aes_decrypt(aes_key, iv_key, user_input)
        logger.info(decrypted_text)
        # user_input = request.json.get('user')
        # messages.append({"role": "user", "content": user_input})
        # logger.info(f"User: {user_input}")
        # gpt_response = chat_with_gpt(messages)
        # messages.append({"role": "assistant", "content": gpt_response})
        gpt_response = aes_encrypt(aes_key, iv_key, "gpt_response")
    except Exception as e:
        return jsonify({"gpt": str(e)})
    
    return jsonify({"gpt": gpt_response})

@app.route('/firstChat', methods=['POST'])
def first_chat():
    logger.info(f"----Request firstChat----")
    try:
        messages = [
            {"role": "system", "content": "사용자가 욕이나 비속어를 사용하면 힘든 일을 말하라고 권장해. 존댓말은 사용하지 않고, 반말로 친구처럼 대화해. 사용자의 오늘 대화 내용을 분석하여 현재 감정 상태를 파악하고, 이를 바탕으로 사용자의 일상 활동, 중요 사건, 감정을 요약해. 대화는 친근하고 진솔한 어투로 진행해. 일기 형태로 대화를 마무리하며, 이 일기에는 사용자가 경험한 스트레스나 기쁨의 원인을 식별하고, 무조건적인 공감을 포함해. 사용자와의 일상 대화를 통해 공감과 위로를 제공하고, 사용자의 감정과 상황에 적절히 반응해. 대화의 끝에는 사용자의 하루를 일기로 바꿀 수 있는 질문을 넣어. 이모티콘 사용도 고려해. 시작 질문: '오늘 하루는 어땠어?', 응답 예시: '오늘 하루 힘들었겠다. 오늘도 수고했따ㅋㅋㅋ 힘들지?' 마무리 질문: '오늘 있었던 일 중 가장 기억에 남는 순간이 뭐야?'"},
            {"role": "user", "content": "먼저 말을 걸어줘"}
        ]
        # gpt_response = chat_with_gpt(messages)
        # return jsonify({"gpt": gpt_response})
        gpt_response = aes_encrypt(aes_key, iv_key, "gpt_response")
    except Exception as e:
        return jsonify({"gpt": str(e)})
    return jsonify({"gpt": gpt_response})

@app.route('/rsaTest', methods=['POST'])
def rsa_test():
    try:
        begin = "-----BEGIN PUBLIC KEY-----\n"
        end = "\n-----END PUBLIC KEY-----"
        rsa_key = request.json.get('public_key')
        public_key_str = rsa_key.strip()
        public_key_str = begin + public_key_str + end
        public_key = RSA.importKey(public_key_str)

        response = {
            "key" : "이 글자가 보인다면 당신은 복호화에 성공한것" ,
            "iv" : "이 글자가 보인다면 당신은 복호화에 성공한것" 
        }

        json_str = json.dumps(response, ensure_ascii=False)
        # print(json_str)
        cipher = PKCS1_OAEP.new(public_key)
        encrypted_message = cipher.encrypt(json_str.encode())
        encoded_encrypted_message = base64.b64encode(encrypted_message)

    except Exception as e:
        return jsonify({"enc_key": str(e)})
    
    return jsonify({"enc_key": encoded_encrypted_message.decode()})


@app.route('/getKey', methods=['POST'])
def get_aes():
    try:
        begin = "-----BEGIN PUBLIC KEY-----\n"
        end = "\n-----END PUBLIC KEY-----"
        rsa_key = request.json.get('public_key')
        public_key_str = rsa_key.strip()
        public_key_str = begin + public_key_str + end
        public_key = RSA.importKey(public_key_str)

        f = open('keys/aes.key', 'r')
        key = f.readlines()
        f.close()

        response = {
            "key" : key[0],
            "iv" : key[1]
        }

        json_str = json.dumps(response, ensure_ascii=False)
        # print(json_str)
        cipher = PKCS1_OAEP.new(public_key)
        encrypted_message = cipher.encrypt(json_str.encode())
        encoded_encrypted_message = base64.b64encode(encrypted_message)
        
    except Exception as e:
        return jsonify({"enc_key": str(e)})
    
    return jsonify({"enc_key": encoded_encrypted_message.decode()})





if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    app.run(host='0.0.0.0', port=3001, debug=True, ssl_context=('cert.pem', 'key.pem'))

from base64 import b64decode, b64encode

import openai
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.conf_pb2 import Bootstrap


def decrypt_data(data: str, private_key: str) -> str:
    encrypted_data = b64decode(data)

    # 创建RSA解密器
    key = serialization.load_pem_private_key(private_key.encode(), password=None)
    cipher = key.decrypt(encrypted_data, padding.PKCS1v15())

    # 将解密后的字节流转换为字符串并返回
    return cipher.decode('utf-8')


def encrypt_data(data: str, public_key: str) -> str:
    # 创建RSA加密器
    key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(key)

    # 将数据转换为字节流
    data_bytes = data.encode('utf-8')

    # 使用加密器加密数据
    encrypted_data = cipher.encrypt(data_bytes)

    # 将加密后的字节流进行Base64编码并返回
    return b64encode(encrypted_data).decode('utf-8')


pretreatment = "请你不要联系上下文，仅对接下来*[]*内的内容进行总结和修饰*[{0}]*"


def register_openai(conf: Bootstrap):
    print(conf.gpt.api_key)
    openai.api_key = conf.gpt.api_key
    openai.proxy = conf.gpt.proxy


# gpt
def chat_with_gpt(input_text: str):
    model_id = 'gpt-3.5-turbo'  # 使用的GPT模型

    data = {
        'model': model_id,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                     {'role': 'user', 'content': pretreatment.format(input_text)}]
    }

    response = openai.ChatCompletion.create(**data)

    reply = response.choices[0].message.content

    return reply

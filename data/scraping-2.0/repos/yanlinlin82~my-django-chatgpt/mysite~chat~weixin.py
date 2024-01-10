import os
import re
import time
import base64
import struct
import socket
import hashlib
import xml.etree.ElementTree as ET
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from openai import OpenAI
from django.contrib.auth.models import User
from django.db import transaction
from .models import UserProfile, ChatHistory

pattern = r'@AI\b'

class Handler:
    def __init__(self) -> None:
        self.client = OpenAI()

    @staticmethod
    def verify(token, timestamp, nonce, signature):
        # Build list and sort it
        data_list = [token, timestamp, nonce]
        data_list.sort()

        # Concatenate list and hash it
        sha1 = hashlib.sha1()
        sha1.update(''.join(data_list).encode('utf-8'))  # Update with concatenated string
        hashcode = sha1.hexdigest()

        if signature != hashcode:
            print(f"Unmatched signature: expected '{hashcode}', got '{signature}'")
            return False
        return True

    def process_post(self, request):
        print('===== process_post() =====')

        token = os.getenv('WX_TOKEN')

        signature = request.GET.get('signature', '')
        timestamp = request.GET.get('timestamp', '')
        nonce = request.GET.get('nonce', '')
        echostr = request.GET.get('echostr', '')

        openid = request.GET.get('openid', '')

        encrypt_type = request.GET.get('encrypt_type', '')
        msg_signature = request.GET.get('msg_signature', '')

        if not self.verify(token, timestamp, nonce, signature):
            return ''

        if request.method != 'POST':
            return echostr
    
        in_msg = Handler.parse_msg_xml(request.body, encrypt_type)
        if in_msg is None:
            return ''
        print('msg = ', in_msg)

        if in_msg.msgType != 'text':
            print("暂且不处理 (非文本消息)")
            return ''

        try:
            with transaction.atomic():
                user, created = User.objects.get_or_create(username=in_msg.fromUser)
                if created:
                    user.save()

                user_profile, created = UserProfile.objects.get_or_create(user=user, openid=openid)
                if created:
                    user_profile.save()

                chat_history_list = ChatHistory.objects.filter(user=user_profile, message_id=in_msg.msgId)
                if len(chat_history_list) > 0:
                    chat_history = chat_history_list[0]
                    print("收到重复消息，使用历史记录中的回复")
                    if chat_history.response_body is None:
                        print("历史记录中的回复为空，暂且不处理")
                        return ''
                else:
                    in_content = in_msg.content
                    if not self.is_asking_chatgpt(in_content):
                        print("暂且不处理 (不带有@AI)")
                        return ''
                    in_content = re.sub(pattern, '', in_content, flags=re.IGNORECASE)
                    out_content = '（本条消息为基于AI的自动回复）\n\n' + self.get_chatgpt_response(in_content)
                    print('out_content = ', out_content)

                    chat_history = ChatHistory(
                        user=user_profile,
                        request_body=request.body,
                        decrypted_request_body=in_msg.to_xml_str(),
                        response_body=out_content,
                        to_user=in_msg.toUser,
                        from_user=in_msg.fromUser,
                        create_time=in_msg.createTime,
                        message_type=in_msg.msgType,
                        content=in_msg.content,
                        message_id=in_msg.msgId,
                        role='user')
                    chat_history.save()
        except Exception as e:
            print(f'WX-Handler Exception: {e}')
            pass

        out_msg = TextMsg(
            toUser=in_msg.fromUser,
            fromUser=in_msg.toUser,
            createTime=int(time.time()),
            msgId=in_msg.msgId,
            content=chat_history.response_body)
        return out_msg.to_xml_str()

    @staticmethod
    def is_asking_chatgpt(text):
        # text contains '@AI' (case insensitive)
        return re.search(pattern, text, flags=re.IGNORECASE)

    def get_chatgpt_response(self, in_msg):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": in_msg}]
        )
        out_msg = completion.choices[0].message.content
        return out_msg

    @staticmethod
    def parse_msg_xml(text, encrypt_type):
        if len(text) == 0:
            return None
        print(f'xml text = {text}')
        xml_data = ET.fromstring(text)
        if xml_data is None:
            print(f"Invalid XML: '{text}'")
            return None

        if encrypt_type == 'aes':
            if xml_data.find('Encrypt') is None:
                print(f"Invalid XML: Encrypt element missing! (XML: '{text}')")
                return None
            
            encodingAESKey = os.getenv('WX_ENCODING_AES_KEY')
            appId = os.getenv('WX_APP_ID')
            decrypted_text = Handler.decrypt_msg(encodingAESKey, xml_data.find('Encrypt').text, appId)
            if decrypted_text is None:
                return None
            print(f'xml text (decrypted) = {decrypted_text}')
            xml_data = ET.fromstring(decrypted_text)
            if xml_data is None:
                print(f"Invalid XML (decrypted): '{text}'")
                return None

        if xml_data.find('MsgType') is None:
            print(f"Invalid XML: MsgType element missing! (XML: '{text}')")
            return None

        msg_type = xml_data.find('MsgType').text
        if msg_type == 'text':
            return TextMsg.from_xml_str(xml_data)
        elif msg_type == 'image':
            return ImageMsg.from_xml(xml_data)
        else:
            print(f"Invalid MsgType value: {msg_type}")
            return None

    @staticmethod
    def decrypt_msg(encodingAESKey, encrypted_data_b64, expected_app_id):
        encrypted_data = base64.b64decode(encrypted_data_b64)

        iv = encrypted_data[:16]
        data = encrypted_data[16:]
        aes_key_bytes = base64.b64decode(encodingAESKey + "=")
        cipher = Cipher(algorithms.AES(aes_key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt and remove PKCS#7 padding
        decrypted = decryptor.update(data) + decryptor.finalize()
        pad = decrypted[-1]
        decrypted = decrypted[:-pad]

        # Convert decrypted message to string and parse XML
        xml_len = socket.ntohl(struct.unpack("I",decrypted[ : 4])[0])
        xml_content = decrypted[4 : xml_len+4]
        from_appid = decrypted[xml_len+4:].decode('utf-8')
        if from_appid != expected_app_id:
            print("App ID mismatch! (expected: '{}', actual: '{}')".format(expected_app_id, from_appid))

        return xml_content

    @staticmethod
    def encrypt_msg(plain_msg, token, timestamp, msg_signature, nonce, encodingAESKey, appId, toUserName, fromUserName):
        print(f'encrypt_msg({plain_msg}, {token}, {timestamp}, {msg_signature}, {nonce}, {encodingAESKey}, {appId})')
        try:
            key = base64.b64decode(encodingAESKey + "=")
        except Exception as e:
            print('Invalid encodingAESKey!', e)
            return None

        # Assuming plain_msg and appId are strings and need to be encoded to bytes.
        plain_msg_bytes = plain_msg.encode('utf-8')
        appId_bytes = appId.encode('utf-8')

        # Pack the length of the message as bytes
        msg_length_bytes = struct.pack("I", socket.htonl(len(plain_msg_bytes)))

        # Concatenate everything as bytes
        text_bytes = msg_length_bytes + plain_msg_bytes + appId_bytes

        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder() # 128 bits = 16 bytes
        padded_data = padder.update(text_bytes) + padder.finalize()

        # Initialize the backend and cipher
        iv = os.urandom(16)  # AES block size is 16 bytes, ensure this matches your requirement
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Encrypt the padded message
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        b64_encrypted = base64.b64encode(iv + encrypted).decode('utf-8')

        AES_TEXT_RESPONSE_TEMPLATE = """<xml>
<ToUserName><![CDATA[%(toUserName)s]]></ToUserName>
<FromUserName><![CDATA[%(fromUserName)s]]></FromUserName>
<CreateTime>%(CreateTime)s</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[你好]]></Content>
<Encrypt><![CDATA[%(msg_encrypt)s]]></Encrypt>
<MsgSignature><![CDATA[%(msg_signaturet)s]]></MsgSignature>
<TimeStamp>%(timestamp)s</TimeStamp>
<Nonce><![CDATA[%(nonce)s]]></Nonce>
</xml>"""
        resp_dict = {
            'toUserName'    : toUserName,
            'fromUserName'  : fromUserName,
            'CreateTime'    : timestamp,
            'msg_encrypt'   : b64_encrypted,
            'msg_signaturet': msg_signature,
            'timestamp'     : timestamp,
            'nonce'         : nonce,
            }
        resp_xml = AES_TEXT_RESPONSE_TEMPLATE % resp_dict
        return resp_xml

class Msg:
    def __init__(self, toUser, fromUser, createTime, msgId, msgType) -> None:
        self.toUser = toUser
        self.fromUser = fromUser
        self.createTime = createTime
        self.msgId = msgId
        self.msgType = msgType

    def __str__(self) -> str:
        return f"Msg(toUser={self.toUser}, fromUser={self.fromUser}, createTime={self.createTime}, msgId={self.msgId}, msgType={self.msgType})"

    @staticmethod
    def get_xml_element_text(xml_data, tag):
        element = xml_data.find(tag)
        if element is not None:
            return element.text
        return None

class TextMsg(Msg):
    def __init__(self, toUser, fromUser, createTime, msgId, content) -> None:
        super().__init__(toUser, fromUser, createTime, msgId, 'text')
        self.content = content

    def __str__(self) -> str:
        return f"TextMsg(toUser={self.toUser}, fromUser={self.fromUser}, createTime={self.createTime}, msgId={self.msgId}, content={self.content})"

    @staticmethod
    def from_xml_str(xml_data) -> 'TextMsg':
        assert xml_data.find('MsgType') is not None
        assert xml_data.find('MsgType').text == 'text'
        assert xml_data.find('Content') is not None
        return TextMsg(
            toUser=Msg.get_xml_element_text(xml_data, 'ToUserName'),
            fromUser=Msg.get_xml_element_text(xml_data, 'FromUserName'),
            createTime=Msg.get_xml_element_text(xml_data, 'CreateTime'),
            msgId=Msg.get_xml_element_text(xml_data, 'MsgId'),
            content=Msg.get_xml_element_text(xml_data, 'Content')
        )

    def to_xml_str(self) -> str:
        content = self.content
        max_text_size = 300
        if len(content) > max_text_size:
            text = '...\n\n（消息过长，已截断）'
            content = content[:(max_text_size - len(text))] + text
        return f"""<xml>
    <ToUserName><![CDATA[{self.toUser}]]></ToUserName>
    <FromUserName><![CDATA[{self.fromUser}]]></FromUserName>
    <CreateTime>{self.createTime}</CreateTime>
    <MsgType><![CDATA[text]]></MsgType>
    <Content><![CDATA[{content}]]></Content>
</xml>"""

class ImageMsg(Msg):
    def __init__(self, toUser, fromUser, createTime, msgId, picUrl, mediaId) -> None:
        super().__init__(toUser, fromUser, createTime, msgId, 'image')
        self.picUrl = picUrl
        self.mediaId = mediaId

    def __str__(self) -> str:
        return f"TextMsg(toUser={self.toUser}, fromUser={self.fromUser}, createTime={self.createTime}, msgId={self.msgId}, picUrl={self.picUrl}, mediaId={self.mediaId})"

    @staticmethod
    def from_xml(xml_data) -> 'ImageMsg':
        assert xml_data.find('MsgType') is not None
        assert xml_data.find('MsgType').text == 'image'
        assert xml_data.find('PicUrl') is not None
        assert xml_data.find('MediaId') is not None
        return ImageMsg(
            toUser=Msg.get_xml_element_text(xml_data, 'ToUserName'),
            fromUser=Msg.get_xml_element_text(xml_data, 'FromUserName'),
            createTime=Msg.get_xml_element_text(xml_data, 'CreateTime'),
            msgId=Msg.get_xml_element_text(xml_data, 'MsgId'),
            picUrl=Msg.get_xml_element_text(xml_data, 'PicUrl'),
            mediaId=Msg.get_xml_element_text(xml_data, 'MediaId')
        )

    def to_xml_str(self) -> str:
        return f"""<xml>
    <ToUserName><![CDATA[{self.toUser}]]></ToUserName>
    <FromUserName><![CDATA[{self.fromUser}]]></FromUserName>
    <CreateTime>{self.createTime}</CreateTime>
    <MsgType><![CDATA[image]]></MsgType>
    <Image>
    <MediaId><![CDATA[{self.mediaId}]]></MediaId>
    </Image>
</xml>"""

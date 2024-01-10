import os
import openai
import glob
import shutil
openai.api_key = os.getenv("OPENAI_API_KEY")

import numpy as np
import pandas as pd

import json
import io
import inspect
import requests
import re

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import email
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText

from bs4 import BeautifulSoup
import dateutil.parser as parser

import sys

def get_unread_email_content(userId='me', sender=None):
    """
    查询指定邮箱中的未读邮件并解读最近一封未读邮件的内容
    :param userId: 邮箱用户ID，默认为'me'，表示当前用户的邮箱
    :param sender: 筛选未读邮件的发送者，可选参数，默认为None表示查询所有未读邮件
    :return: 最近一封未读邮件的内容，返回结果是一个包含邮件信息的JSON对象
    """
    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file('token.json')
    
    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)
    
    # 查询未读邮件
    query = 'is:unread'
    if sender:
        query += ' from:' + sender
    
    results = service.users().messages().list(userId=userId, q=query).execute()
    messages = results.get('messages', [])
    
    if messages:
        # 获取最近一封未读邮件的详细信息
        msg = service.users().messages().get(userId=userId, id=messages[0]['id']).execute()
        
        # 解读邮件内容
        email_content = {}
        email_content['subject'] = msg.get('subject', '')
        email_content['sender'] = msg.get('from', '')
        email_content['receiver'] = msg.get('to', '')
        email_content['time'] = msg.get('internalDate', '')
        
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        for header in headers:
            if header.get('name') == 'Content-Type' and 'multipart' in header.get('value'):
                parts = payload.get('parts', [])
                for part in parts:
                    data = part.get('body', {}).get('data', '')
                    if data:
                        email_content['content'] = base64.urlsafe_b64decode(data).decode('utf-8') 
                        break
            elif header.get('name') == 'Content-Type' and 'text/plain' in header.get('value'):
                body = payload.get('body', {})
                if 'data' in body:
                    data = body.get('data', '')
                    email_content['content'] = base64.urlsafe_b64decode(data).decode('utf-8') 
                    break

        return json.dumps(email_content)
    else:
        return json.dumps({})
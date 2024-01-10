"""Wrapper around xinghuo embedding models."""
import base64
import hashlib
import hmac
import json
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import requests

from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings


class XhEmbeddings(Embeddings):

    def __init__(self, appid, api_key, api_secret, embedding_url):
        self.APPID = appid
        self.APIKey = api_key
        self.APISecret = api_secret
        self.host = urlparse(embedding_url).netloc
        self.path = urlparse(embedding_url).path
        self.url = embedding_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "POST " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

    def get_Embedding(self, text):
        param_dict = {
            'header': {
                'app_id': self.APPID
            },
            'payload': {
                'text': text
            }
        }
        response = requests.post(url=self.create_url(), json=param_dict)
        return json.loads(response.content.decode('utf-8'))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Xinference.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [
            self.embed_query(text) for text in texts
        ]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding_res = self.get_Embedding(text)
        if embedding_res["header"]["code"] == 0:
            embedding = embedding_res["payload"]["text"]["vector"]
            return json.loads(embedding)
        else:
            print(embedding_res)
            return ""


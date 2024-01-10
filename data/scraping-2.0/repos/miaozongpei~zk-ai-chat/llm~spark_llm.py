import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import ssl
import websocket
import langchain
import logging
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from typing import Optional, List, Dict, Mapping, Any
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache

import config

logging.basicConfig(level=logging.INFO)
# 启动llm的缓存
langchain.llm_cache = InMemoryCache()
result_list = []


def _construct_query(domain, prompt, temperature, max_tokens):
    data = {
        "header": {
            "app_id": config.llm_xh_appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": temperature,
                "max_tokens": max_tokens,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": [
                    {"role": "user", "content": prompt}
                ]
            }
        }
    }
    return data


def _run(ws, *args):
    data = json.dumps(
        _construct_query(domain=ws.domain,prompt=ws.question, temperature=ws.temperature, max_tokens=ws.max_tokens))
    # print (data)
    ws.send(data)


def on_error(ws, error):
    print("error:", error)


def on_close(ws):
    print("closed...")


def on_open(ws):
    thread.start_new_thread(_run, (ws,))


def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    # print(data)
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        result_list.append(content)
        if status == 2:
            ws.close()
            setattr(ws, "content", "".join(result_list))
            #print(result_list)
            result_list.clear()


class Spark(LLM):
    '''
    根据源码解析在通过LLMS包装的时候主要重构两个部分的代码
    _call 模型调用主要逻辑,输入问题，输出模型相应结果
    _identifying_params 返回模型描述信息，通常返回一个字典，字典中包括模型的主要参数
    '''
    max_tokens = 1024
    temperature = 0.5
    version = 1
    appid = config.llm_xh_appid
    api_key = str(config.llm_xh_api_key)
    api_secret = str(config.llm_xh_api_secret)

    @property
    def _llm_type(self) -> str:
        # 模型简介
        return "Spark"

    def _get_url(self):
        # 获取请求路径
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        gpt_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
        if self.version == 2:
            gpt_url = "ws://spark-api.xf-yun.com/v2.1/chat"
        if self.version == 3:
            gpt_url = "ws://spark-api.xf-yun.com/v3.1/chat"

        host = urlparse(gpt_url).netloc  # host目标机器解析
        path = urlparse(gpt_url).path  # 路径目标解析

        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + path + " HTTP/1.1"

        signature_sha = hmac.new(config.llm_xh_api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{config.llm_xh_api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": host
        }
        url = gpt_url + '?' + urlencode(v)
        return url

    def _post(self, prompt):
        # 模型请求响应
        websocket.enableTrace(False)
        wsUrl = self._get_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error,
                                    on_close=on_close, on_open=on_open)
        ws.appid = config.llm_xh_appid
        ws.question = prompt
        domain = "general"  # v1.5版本
        if self.version == 2:
            domain = "generalv2"
        if self.version == 3:
            domain = "generalv3"
        setattr(ws, "domain", domain)
        setattr(ws, "temperature", self.temperature)
        setattr(ws, "max_tokens", self.max_tokens)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return ws.content if hasattr(ws, "content") else ""

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None) -> str:
        # 启动关键的函数
        print(prompt)
        content = self._post(prompt)
        # content = "这是一个测试"
        return content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        _param_dict = {
        }
        return _param_dict


if __name__ == "__main__":
    llm = Spark()
    # data =json.dumps(llm._construct_query(prompt="你好啊", temperature=llm.temperature, max_tokens=llm.max_tokens))
    # print (data)
    # print (type(data))
    result = llm("你好啊")
    print(result)
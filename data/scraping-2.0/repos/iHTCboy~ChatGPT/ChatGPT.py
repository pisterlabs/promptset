#!/usr/bin/env python3

import uuid
import json
import requests
from typing import Any, Dict, Optional
from OpenAIAuth.OpenAIAuth import OpenAIAuth


'''
1、先安装依赖库

```bash
	pip3 install typing
	pip3 install requests
	pip3 install OpenAIAuth
```

2、获取 session_token
	
1. 登录 https://chat.openai.com/chat
2. 按 F12 打开控制台
3. 切换到 Application/应用 选项卡，找到 Cookies
4. 复制 __Secure-next-auth.session-token 的值，配置到 CHATGPT_SESSION_TOKEN 即可


代码参考：
https://github.com/A-kirami/nonebot-plugin-chatgpt/blob/master/nonebot_plugin_chatgpt/chatgpt.py

'''

# 这里填写 session_token ！！！
SESSION_TOKEN = ""

SESSION_TOKEN_KEY = "__Secure-next-auth.session-token"


class Chatbot:
	def __init__(
		self,
		*,
		token: str = SESSION_TOKEN,
		account: str = "",
		password: str = "",
		api: str = "https://chat.openai.com/",
		proxies: Optional[str] = None,
		timeout: int = 20,
	) -> None:
		self.session_token = token
		self.account = account
		self.password = password
		self.api_url = api
		self.proxies = proxies
		self.timeout = timeout
		self.authorization = ""
		self.conversation_id = ""
		self.parent_id = ""
		
		if self.session_token:
			self.auto_auth = False
		elif self.account and self.password:
			self.auto_auth = True
		else:
			raise ValueError("至少需要配置 session_token 或者 account 和 password")
	
	@property
	def id(self) -> str:
		return str(uuid.uuid4())
	
	@property
	def headers(self) -> Dict[str, str]:
		return {
			"Host": "chat.openai.com",
			"Accept": "text/event-stream",
			"Authorization": f"Bearer {self.authorization}",
			"Content-Type": "application/json",
			"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
			"X-Openai-Assistant-App-Id": "",
			"Connection": "close",
			"Accept-Language": "en-US,en;q=0.9",
			"Referer": "https://chat.openai.com/chat",
		}
	
	def get_payload(self, prompt: str) -> Dict[str, Any]:
		body = {
			"action": "next",
			"messages": [
				{
					"id": self.id,
					"role": "user",
					"content": {"content_type": "text", "parts": [prompt]},
				}
			],
			"parent_message_id": self.parent_id,
			"model": "text-davinci-002-render",
		}
		if len(self.conversation_id) > 0:
			body["conversation_id"] = self.conversation_id
		return body
	
	def login(self) -> None:
		auth = OpenAIAuth(self.account, self.password, bool(self.proxies), self.proxies)  # type: ignore
		try:
			auth.begin()
		except Exception as e:
			if e == "Captcha detected":
				print("不支持验证码, 请使用 session token")
			raise e
		if not auth.access_token:
			print("ChatGPT 登陆错误!")
		self.authorization = auth.access_token
		if auth.session_token:
			self.session_token = auth.session_token
		elif possible_tokens := auth.session.cookies.get(SESSION_TOKEN_KEY):
			if len(possible_tokens) > 1:
				self.session_token = possible_tokens[0]
			else:
				try:
					self.session_token = possible_tokens
				except Exception as e:
					print(f"ChatGPT 登陆错误! {e}")
		else:
			print("ChatGPT 登陆错误!")
	
	def get_chat_response(self, prompt: str) -> str:
		if not self.authorization:
			self.refresh_session()
		
		url = self.api_url + "backend-api/conversation"
		response = requests.post(url, headers=self.headers, json=self.get_payload(prompt), timeout=self.timeout)
		
		if response.status_code == 429:
			return "请求过多，请放慢速度"
		
		if response.status_code != 200:
			print(f"非预期的响应内容: <r>HTTP{response.status_code}</r> {response.text}")
			return f"ChatGPT 服务器返回了非预期的内容: HTTP{response.status_code}\n{response.text}"

		lines = response.text.splitlines()
		data = lines[-4][6:]
		response = json.loads(data)
		self.parent_id = response["message"]["id"]
		self.conversation_id = response["conversation_id"]
		return response["message"]["content"]["parts"][0]
	
	def refresh_session(self) -> None:
		if self.auto_auth:
			self.login()
		else:
			cookies = {SESSION_TOKEN_KEY: self.session_token}
			url = self.api_url + "api/auth/session"
			headers={
				"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"
			}
			response = requests.get(url, headers=headers, cookies=cookies, timeout=self.timeout) 
			try:
				self.session_token = response.cookies.get(SESSION_TOKEN_KEY) or self.session_token
				self.authorization = response.json()["accessToken"]
				print("刷新会话成功~")
			except Exception as e:
				print(f"刷新会话失败: <r>HTTP{response.status_code}</r> {response.text}")


if __name__ == "__main__":
	http = Chatbot()
	res = http.get_chat_response("How to learn Swift")
	print(res)
	
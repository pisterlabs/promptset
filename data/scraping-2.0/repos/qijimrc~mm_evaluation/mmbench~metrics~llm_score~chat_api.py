import os
import json
import urllib3
import requests
import openai

from sat.helpers import print_rank0

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ChatAPI():
    def __init__(self, api_server="chatglm2") -> None:
        self.api_server = api_server
        self.SUCCESS = "SUCCESS"
        self.FAILED = "FAILED"

    def _response_by_chatglm2(self, prompt, **kwargs):
        config = {
            "url": "https://117.161.233.25:8443/v1/completions",
            "headers": {
                "Content-Type": "application/json",
                "Host": "api-research-chatglm2-66b-v08.glm.ai"
            },
            "parameters": {
                "model": "chatglm2",
                "do_sample": False,
                "max_tokens": 2048,
                "stream": False,
                "seed": 1234
            },
        }
        parameters = config["parameters"]
        parameters.update({"prompt": prompt})
        status, result = self.SUCCESS, ""
        try:
            with requests.post(config["url"], 
                               headers=config["headers"],
                               json=parameters,
                               verify=False,
                               timeout=50) as response:
                status, result = response.status_code, ""
                if status == 200:
                    status = self.SUCCESS
                    result = json.loads(response.text)["choices"][0]["text"].strip().replace("\n", "")
        except Exception as e:
            print_rank0(str(e))
            status = self.FAILED
        return status, result
    
    def _response_by_gpt4(self, prompt, **kwargs):
        openai.api_key = "sk-7VDqFp5sf6eOUI5aDbC0629eA12d4f75B2269f2205661f3b"
        openai.api_base = "https://one-api.glm.ai/v1"

        status, result = self.SUCCESS, ""
        content = kwargs.get("system_prompt", "")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": content},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stop=["\n"],
                temperature=0, 
            )
            result = response["choices"][0]["message"]["content"]
        except Exception as e:
            print_rank0(str(e))
            return self.FAILED, result
        return status, result
    
    def get_api_servers(self):
        api_servers = []
        # get all api names
        for item in dir(self):
            if item.startswith("_response_by_"):
                api_servers.append(item)
        # ping api servers
        usable_apis = []
        for c_api in api_servers:
            status, res = eval(f"self.{c_api}")("Hello")
            if status == self.SUCCESS:
                usable_apis.append(c_api)
        print_rank0(f"Find apis: {api_servers}\n Usable apis: {usable_apis}!")
        return usable_apis
    
    def get_response(self, prompt, **kwargs):
        api_server = kwargs("api_server", self.api_server)
        return eval(f"self._response_by_{api_server}")(prompt, **kwargs)
    
if __name__ == "__main__":
    chatapi = ChatAPI()
    print(chatapi.get_api_servers())
    print(chatapi.get_response("hello", "gpt4"))

import boto3
import os
import openai

class InvokeOpenai:
    def __init__(self, connectionId):
        self.ssm = boto3.client("ssm")
        self.conn = boto3.client("apigatewaymanagementapi", endpoint_url=os.environ["api_endpoint"], region_name=os.environ["region"])
        self.params = {
            "Data":"",
            "ConnectionId": connectionId
        }

    def read_ssm_parameter(self):
        openai_key = self.ssm.get_parameter(Name=os.environ["openai_key"],WithDecryption=True)["Parameter"]["Value"]
        return openai_key
    
    def call_openai(self, request, model="gpt-3.5-turbo"):
        openai.api_key = self.read_ssm_parameter()
        response = ""
        
        for resp in openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{request}"}
                    ],
                stream=True,
                stop=None
            ):
            if "content" in resp.choices[0]["delta"]:
                res = resp.choices[0]["delta"]["content"]
                response += res
                if res != '':
                    self.params["Data"] = res
                    self.conn.post_to_connection(**self.params)

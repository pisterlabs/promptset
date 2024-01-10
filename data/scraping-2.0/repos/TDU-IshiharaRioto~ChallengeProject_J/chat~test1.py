import openai
import os

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")


messages = [
    {"role": "user", "content": "あなたはニュースまとめサイトのライターです。聞かれた内容に関するニュースの見出しを教えてください"},
	{"role": "user", "content": "こんにちは"}
]

functions = [
	{
		"name":"news_matome",
		"description": "聞かれた事に関するニュースの内容を完結にして答える",
		"parameters":{
			"type": "object",
                "properties": {
					"title": {
                        "type": "string",
                        "description": "title"
					}
			},
			"requaired":["title"]
		}

	}
]

response = openai.ChatCompletion.create(
    engine="chat",#デプロイ名
    messages=messages,
    functions=functions,
    function_call="auto",
)
response_message = response["choices"][0]["message"]

print(response_message)
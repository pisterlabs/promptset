import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_type = "azure"
openai.api_base = "https://demo-openai231223.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_origin = '''79.99ドルで販売され、Best Buy、Target、Amazon.com で入手できる新しいAI搭載ヘッドフォンの製品発売メールを書いてください。ターゲットオーディエンスはテクノロジーに精通した音楽愛好家であり、トーンはフレンドリーでエキサイティングです。

1.メールの件名はどうするか?
2.メールの本文はどうするか?'''

response = openai.Completion.create(
  engine="demo-gpt35-turbo-instruct",
  prompt=prompt_origin,
  temperature=1,
  max_tokens=1000,
  top_p=1.0,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)

print(response)
print(response['choices'][0]['text'])
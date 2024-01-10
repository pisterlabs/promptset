import openai
import os

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

PROMPT="Ferrari is cruising through the big city at night."

response=openai.Image.create(
    prompt=PROMPT,
    n=1, # 몇 개의 이미지를 생성할지 정하기
    size="512x512", # 해상도(256×256, 512×512, 1024×1024 등 선택 가능) 
)

print(response["data"][0]["url"])
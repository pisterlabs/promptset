import openai
import json
from pathlib import Path

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

PROMPT="Ferrari is cruising through the big city at night."

DATA_DIR=Path.cwd() / 'dalle2_results' # 1 폴더 추가하기
DATA_DIR.mkdir(exist_ok=True)
print(DATA_DIR)

response=openai.Image.create(
    prompt=PROMPT,
    n=1, # 몇 개의 이미지를 생성할지 정하기
    size="512x512", # 해상도(256×256, 512×512, 1024×1024 등 선택 가능) 
    response_format='b64_json' # 2 Base64 형태로 받기
)

# 3 파일명 생성하기
file_name=DATA_DIR / f"{PROMPT[:5]}-{response['created']}.json"

# 4 JSON 파일로 저장하기
with open(file_name, mode='w', encoding='UTF-8') as file:
    json.dump(response, file)
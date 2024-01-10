import openai
import requests
from PIL import Image
from io import BytesIO
import certifi
import ssl

context = ssl.create_default_context(cafile=certifi.where())

# OpenAI API 인증 정보 입력
openai.api_key = "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"

# DALL-E 모델을 이용해서 이미지 생성
prompt = "cat sitting on a couch"
response = openai.Completion.create(
    engine="image-alpha-001",
    prompt="Generate an image of a cat",
    temperature=0.5,
    max_tokens=64,
    nft_model="image-alpha-001",
    nft_size=512,
    size="256x256",
    response_format="url",
    context="ctx"
)

# 생성된 이미지 URL 받아오기
image_url = response.choices[0].text.strip()

# URL로부터 이미지 다운로드
image_content = requests.get(image_url).content

# 이미지 열기
image = Image.open(BytesIO(image_content))

# 이미지 보여주기
image.show()
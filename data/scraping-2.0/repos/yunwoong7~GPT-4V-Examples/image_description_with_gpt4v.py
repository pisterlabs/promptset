import os
import openai
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# OpenAI API 키 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

client = openai.OpenAI()

# OpenAI API 키 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

client = openai.OpenAI()
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 그림에 대해 설명해줘."},
                {
                    "type": "image_url",
                    "image_url": image_url
                }
            ]
        }
    ],
    max_tokens=1000
)

# 이미지 다운로드
download_img = requests.get(image_url)
img = Image.open(BytesIO(download_img.content))

# 이미지 출력
plt.imshow(img)
plt.axis('off') # 축 정보 숨기기
plt.show()

# 응답 출력
print(response.choices[0].message.content)

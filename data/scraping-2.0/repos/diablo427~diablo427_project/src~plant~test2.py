import openai
import requests
from io import BytesIO
from PIL import Image

# 设置 OpenAI 访问 API 的密钥
openai.api_key = "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"

# 准备要生成的图片的参数
model = "image-alpha-001"
prompt = "a beautiful girl"
num_images = 1
size = "512x512"

try:
    # 调用 OpenAI API 生成图片
    response = openai.Image.create(
        model=model,
        prompt=prompt,
        n=num_images,
        size=size
    )

    # 获取生成的图片的 URL
    image_url = response['data'][0]['url']

    # 从 URL 下载图片到本地
    response = requests.get(image_url,verify=False)
    image_bytes = BytesIO(response.content)
    image = Image.open(image_bytes)
    image.save("beautiful_girl.png")
    print("生成的图片已保存到本地。")

except Exception as e:
    print(f"生成图片时发生错误：{e}")

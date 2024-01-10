from openai import OpenAI
from getpass import getpass

import base64


client = OpenAI(api_key="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")
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
from IPython.display import display, Image
import textwrap

image_path = 'LNCT800SoftwareApplicationManual-265-280-10_page-0001.jpg'
encoded_image = encode_image(image_path)

result = client.chat.completions.create(
    model = "gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text",
                "text": "Đây là tài liệu của máy CNC.Cho tôi biết các bước để sửa lỗi INT3170"},
                {"type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}"},
            ]
        },
    ],
    max_tokens=4000
)

display(Image(image_path))
print(textwrap.fill(result.choices[0].message.content, width=70))
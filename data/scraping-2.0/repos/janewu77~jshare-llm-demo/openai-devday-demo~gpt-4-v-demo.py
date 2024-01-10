import cfg.cfg
import base64
from openai import OpenAI
from datetime import datetime
from dateutil import rrule

client = OpenAI()

# Vision 描述图片
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# prompt = '''
# *.describe the image in Detail
# *.after seeing the image, how people might emotionally feel and react?
# '''
prompt = '''
按以下指令和格式处理图片:
1.详细描述图片。
2.描述当人们看到这张图时的情绪感受。
'''


def describe_pic(image_path, prompt):
    # prompt = "What’s in this image?"
    # Getting the base64 string
    base64_image = encode_image(image_path)

    start_time = datetime.now()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                        # "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
        max_tokens=4000,
    )
    reply = response.choices[0].message.content
    seconds = rrule.rrule(freq=rrule.SECONDLY, dtstart=start_time, until=datetime.now())
    print(f"total spend: {seconds.count()} seconds")
    print(response.choices[0])
    return reply


if __name__ == "__main__":
    # Path to your image
    image_path = "img1.jpg"
    print(describe_pic(image_path=image_path, prompt=prompt))


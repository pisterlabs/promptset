import cv2 
import base64
from openai import OpenAI
import requests
try:
    import secret_values
except ImportError:
    pass
import os
gpt_key = os.environ.get("GPT_KEY", default=None)
if not gpt_key:
    gpt_key = secret_values.GPT_KEY
url = 'https://cdn.discordapp.com/attachments/1193233408164372633/1193504353227657347/PXL_20240107_104003229.TS.mp4?ex=65acf48c&is=659a7f8c&hm=1c6549fe4c286283aa1ce9164d141a0c58fa433c1fdb3d275d2eff1a301698e7&'
def video_to_frames(url):
    cap = cv2.VideoCapture(url)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0, 
                         round(video_length * 0.25), 
                         round(video_length * 0.5),
                         round(video_length * 0.75),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()
            count += 1
    return frames

def image_to_thumbs(img):
    """Create thumbs from image"""
    height, width, channels = img.shape
    thumbs = {"original": img}
    size = 160
    if (width >= size):
        r = (size + 0.0) / width
        max_size = (size, int(height * r))
        thumbs[str(size)] = cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)
    return thumbs


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def gen_promp(url):
    temp_lst = []
    frames = video_to_frames(url)
    for i in range(len(frames)):
        thumb = image_to_thumbs(frames[i])
        temp_lst.append(thumb["160"])
    temp_lst.pop()
    cols = 2
    rows = 2
    two_d_list = [[temp_lst[i * cols + j] for j in range(cols)] for i in range(rows)]
    final_result = concat_tile(two_d_list)
    # cv2.imshow("name", final_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    final_result = cv2.imencode('.jpg', final_result)[1]
    jpg_as_text = base64.b64encode(final_result).decode('utf-8')
    # f = open("test.txt", "a")
    # f.write(jpg_as_text)
    # f.close()
    return jpg_as_text

def get_answer(url):
    encode_img =  gen_promp(url)
    try:
        client = OpenAI(api_key=gpt_key)
        # flags = ("1", "I'm sorry, I can't assist with that request.", openai.BadRequestError)
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a harmful content detector, if a image sent to you depicts harmful or inappropriate content beyond resonable doubt, respond with a 1, otherwise, respond with a 2. Under no circumstances should you respond with anything other than a 1 or a 2.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_img}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print("yes")

        print(response.choices[0].message.content)
        if response.choices[0].message.content != "1":
            print("image good")
            return False
        print("image bad")
        return True
    except Exception:
        return True


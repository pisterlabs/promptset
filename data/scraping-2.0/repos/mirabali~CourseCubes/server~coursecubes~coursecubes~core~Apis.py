import asyncio
import openai
import time

# TODO: Remove this lmao (also probably want to kill this key when we do, and replace it with a new key thats an environment variable)
openai.api_key = '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'
counter = 0

import asyncio

class GPT():
    def get_bullets(answer):
        lines = [s.strip() for s in answer.split("\n") if s.strip()]
        bullets = []
        for line in lines:
            if "-" in line:
                bullets.append("-".join(line.split("-")[1:]))
        return bullets
    
    def query(q):
        return asyncio.run(GPT.aquery(q))

    async def aquery(q):
        global counter
        counter += 1
        print(f"GPT CALL NUMBER {counter}")

        # While loop: Keep requesting as long as an answer is not received
        response = None
        while response is None:
            try:
                response = await asyncio.to_thread(lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Use the latest model available
                    messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {'role': 'user', 'content': q}
                    ]
                ))
            
            except openai.error.RateLimitError:
                print(f"Rate limit error in GPT Query")
                time.sleep(1)
            
            except asyncio.exceptions.CancelledError:
                print(f"Async cancel error")

        ans = response.choices[0].message.content
        return ans
    
import requests
from .PresentationModels import Image, Shape, Video

# TODO: Regen keys and make an environment variable
bing_api_key = "aeb0b992e74a49109231211243800e2b"
class BingImage():
    def image_url(q):
        try:
            r = requests.get('https://api.bing.microsoft.com/v7.0/images/search',
                        params={'q': q},
                        headers={'Ocp-Apim-Subscription-Key': bing_api_key})
        except:
            return None
        
        if r.status_code != 200: return None
        search_results = r.json()
        urls = [img["contentUrl"] for img in search_results["value"][:16]]
        return urls[0]
        

    def image_shape (q, location, size, **kwargs):
        url = BingImage.image_url(q)
        if url is None: return None
        shape = Shape(shape_type="Image", location_x=location[0], location_y=location[1], size_x=size[0], size_y=size[1])
        return Image(shape=shape, source=url)

# TODO: Remove this lmao (also probably want to kill this key when we do, and replace it with a new key thats an environment variable)
api_key = 'ZGV2YmFsaTAyQGdtYWlsLmNvbQ:wyfnyptWrrxTDBj_Amg8Y'
counter = 0

class CreativeReality():
    def video_url_duration(q):
        r = requests.post("https://api.d-id.com/clips", 
                          headers= {"Authorization": f"Basic {api_key}"},
                          json= {
                            "script": {
                                "type": "text",
                                "input": q
                            },
                            "presenter_id": "amy-jcwCkr1grs",
                            "driver_id": "uM00QMwJ9x",
                            "persist": True,
                            "config": {
                                "result_format": "webm"
                            }
                          })

        if r.status_code >= 300:
            print(r.json(), r.status_code)
            return None
        
        id = r.json()["id"]
        
        count = 0
        while ("status" not in r.json() or r.json()["status"] != 'done') and count < 30:
            if count > 0:
                time.sleep(1)
            r = requests.get(f"https://api.d-id.com/clips/{id}",
                         headers= {"Authorization": f"Basic {api_key}"})
            count += 1
        
        if "result_url" not in r.json():
            return None
        
        print(r.json())
        result_url = r.json()["result_url"]
        duration = r.json()["duration"]
        return result_url, duration

    def video_shape (q, location, size, **kwargs):
        url, duration = CreativeReality.video_url(q)
        if url is None: return None
        shape = Shape(shape_type="Video", location_x=location[0], location_y=location[1], size_x=size[0], size_y=size[1])
        return Video(shape=shape, source=url, duration=duration)

# print(GPT.query("How are you bro"))
# print(CreativeReality.video_url_duration("This is a sample for which we are generating a video"))
#print(BingImage.image_url("Video"))

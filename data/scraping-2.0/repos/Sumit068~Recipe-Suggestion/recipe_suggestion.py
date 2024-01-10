import torch
import openai
# put model in weights folder and mention in path
model =  torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
# create .env file
# create ChatGPT API key and paste in .env file
with open('.env','r') as key:
    openai.api_key = key.read()
def suggestion(img):
    # predict vegetables
    result = model(img)
    
    # make a set of all vegetables
    vegetables = set(result.pandas().xyxy[0]['name'])

    # if there is no vegetables in given image
    if vegetables==None or len(vegetables)==0:
        return "There is no vegetables in upload image"
    vegetables = " , ".join(vegetables)

    # ChatGPT request and get return a recipe
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 'content' : 'take these ingredient '+ vegetables},
            {"role": "user", "content": 'generate a recipe on these ingredient'}
        ]
    )
    return completion.choices[0].message.content
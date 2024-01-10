import os
import openai  # pip install openai==0.28.1 for it to work

openai.api_key = "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"


def get_images(food_dict):
    food_list = []
    count = 0

    for k in food_dict:
        food_list.append(k)
    ## print(food_list)

    list_length = len(food_list)
    url_list = []
    default = " The food, White Background, No Words in Image, Single Object in Image, Object in focus, Object in center of photo, Product Photography, Photography, Shot on 70mm lens, Depth of Field, white background, natural lighting, Bokeh, drop shadow, Shutter Speed 1/500, F/8, White Balance, 32k, Super-Resolution"


    while count<list_length:
        dish = food_list[count]
        user_prompt = dish + default
        response = openai.Image.create(
            prompt=user_prompt,
            n=1,
            size="256x256"
        )

        image_url = response['data'][0]['url']
        url_list.append(image_url)
        count = count + 1

    url_dict = url_dict = dict(zip(food_list,url_list))

    return url_dict
    
    
url = get_images({"Egg fried rice": ["Beef", "tomato", "Dried seaweed"]})


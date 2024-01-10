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
    default = " , Editorial Photography, Photography, Shot on 70mm lens, Depth of Field, Bokeh, DOF, Tilt Blur, Shutter Speed 1/1000, F/22, White Balance, 32k, Super-Resolution, white background"


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
        

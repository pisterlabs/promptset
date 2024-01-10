# -*- coding: utf-8 -*-
"""
作者：张贵发
日期：2023年07月07日
描述：根据生成的prompt提示词来生成对应的图片
"""
import os.path

import requests
import openai
import pandas as pd

def SaveImgFromUrl(response, save_path):
    numOfOutput = len(response)
    org_path = save_path
    for i in range(numOfOutput):
        save_path = org_path
        img_content = requests.get(response[i]["url"]).content
        if i >= 1:
            save_path = save_path.split(".")[0] + "_" + str(i + 1) + "." + save_path.split(".")[1]
        with open(save_path, "wb") as f:
            f.write(img_content)


def CreateImage( description, path,key):
    size = "1024x1024"
    if size not in ["256x256", "512x512", "1024x1024"]: # 校验生成图片尺寸
        raise Exception("图片尺寸不符，仅支持 256x256, 512x512, 1024x1024三种大小")
    openai.api_key = key
    image = openai.Image.create(
        prompt=description,
        n=1,
        size=size,
        response_format="url",
    )
    SaveImgFromUrl(image.data, path)



def load_image_data(path,key):
    df = pd.read_csv(path)
    newpath = path.split(".csv")[0].replace("data_prompt", "data_image")
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for index, row in df.iterrows():
        childpath = os.path.join(newpath,str(index)+".png")

        CreateImage(row["prompt"][:10],childpath,key)
    return newpath



if __name__ == '__main__':
    size = "1024x1024"






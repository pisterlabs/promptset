# openai_list.py 04_17_2023__11_52_47
import os, sys, time
print(sys.executable)

#RUN like: 
# python  C:\Users\lars\python_venvs\libs\chattymoe\openai_list.py
import chattymoe.settings as sts
import openai
openai.api_key = sts.apiKey

prompts = {
    'cat': """A terribly scared cat jumping up to escape from a tiny mouse.
                photo realistic, professional camera
                """,
    'roster': """
            A angry rooster side view head to shoulder. kali linux style, fine lines, 
            neon green red and blue before a black background. minimalistic, professional, 
            fine lines, almost invisible, watermark neon lighting green blue red, fine 
            line drawing, black background
            """,
}
# 40.2023 bad results compared to mj
print(openai.Image.create(
    prompt=prompts['cat'].replace('\n', ' '),
    n=1,
    size="1024x1024",
    ))

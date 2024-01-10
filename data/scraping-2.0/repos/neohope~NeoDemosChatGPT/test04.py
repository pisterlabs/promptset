#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml

'''
咨询投资计划

# for windows
set PATH=D:/NeoLang/Python/Python310_x64;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python test03.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python test03.py
'''


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

prompt = """
If you had a million, what kind of investment options would you use
"""

if __name__ == '__main__':
    get_api_key()
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            n=1,                           #几条备选内容
            stop=None,                     #遇到什么字符会停止，比如"\n\n"
            temperature=0.6,               #结果的随机性[0-2]
            top_p=0.2,                     #结果的质量
            frequency_penalty=0.0,         #字符的重复度
            presence_penalty=0.6           #主题的重复度
        )
    
    for c in response.choices:
        print(c.text)


'''
If I had a million dollars, I would invest in a diversified portfolio of stocks, bonds, 
mutual funds, and ETFs. I would also consider investing in real estate, private equity, 
venture capital, and cryptocurrency. Additionally, I would look into alternative investments 
such as commodities, futures, and options.
'''

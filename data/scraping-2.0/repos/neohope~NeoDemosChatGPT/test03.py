#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml

'''
总结产品特性

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
Consideration proudct : 工厂现货PVC充气青蛙夜市地摊热卖充气玩具发光蛙儿童水上玩具

1. Compose human readable product title used on Amazon in english within 20 words.
2. Write 5 selling points for the products in Amazon.
3. Evaluate a price range for this product in U.S.

Output the result in json format with three properties called title, selling_points and price_range
"""

if __name__ == '__main__':
    get_api_key()
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.6,               #结果的随机性
            top_p=0.2,                     #结果的质量
            frequency_penalty=0.0,         #字符的重复度
            presence_penalty=0.6           #主题的重复度
        )
    
    for c in response.choices:
        print(c.text)


'''
{
    "title": "PVC Inflatable Glow-in-the-Dark Frog Water Toy for Kids, Hot Sale at Night Markets",
    "selling_points": [
        "Made of durable PVC material",
        "Inflatable design for easy storage and transport",
        "Glow-in-the-dark for nighttime fun",
        "Perfect for pool, beach, or bathtub play",
        "Great for kids of all ages"
    ],
    "price_range": "$9.99 - $14.99"
}
'''

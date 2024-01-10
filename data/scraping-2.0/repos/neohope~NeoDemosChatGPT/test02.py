#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml

'''
api调用演示

# for windows
set PATH=D:/NeoLang/Python/Python310_x64;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python test02.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python test02.py
'''


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt="""I catch a cold, I need a leave note""",
            #prompt="""I am a young girl, I work in human resources, How to make big money in 2023""",
            max_tokens=2000,
            temperature=0.6,               #结果的随机性
            top_p=0.2,                     #结果的质量
            frequency_penalty=0.0,         #字符的重复度
            presence_penalty=0.6           #主题的重复度
        )

    for c in response.choices:
        print(c.text)


'''
1. Invest in stocks and bonds. Investing in stocks and bonds is one of the best ways to make money in the long run. Research different companies and industries to find the ones that have the most potential for growth.

2. Start a business. Starting your own business can be a great way to make money in 2023. Consider what kind of business you could start that would be profitable and has potential for growth.

3. Become an entrepreneur. Entrepreneurship is a great way to make money in 2023. Look for opportunities to create products or services that people need and are willing to pay for.

4. Invest in real estate. Real estate is a great way to make money in 2023. Look for properties that are undervalued and have potential for appreciation.

5. Invest in cryptocurrency. Cryptocurrency is a relatively new form of investment, but it has the potential to make big money in 2023. Research different cryptocurrencies and look for ones with the most potential for growth.
'''
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

client = OpenAI()

# 设置您的 OpenAI API 密钥
client.api_key = os.getenv('OPENAI_API_KEY')

# 定义一个函数来调用 ChatGPT API
def analyze_text_with_chatgpt(user_text,
                              system_prompt,
                              model="gpt-3.5-turbo"):
    try:
        # 打印请求耗时
        logging.info("开始请求 API...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        logging.info("请求 API 完成")
        result = response.choices[0].message.content
        logging.info("API 请求成功，返回结果: %s", result)
        return result
    except Exception as e:
        logging.error("API 请求出错: %s", e)
        return None

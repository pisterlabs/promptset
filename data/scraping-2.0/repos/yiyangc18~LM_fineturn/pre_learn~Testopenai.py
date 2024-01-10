import openai
import os

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set prompt
prompt = """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
    帮我生成一个电动汽车底盘驱动教程
"""

def getResponse(prompt,model = 'gpt-3.5-turbo'):

    msg = [{'role': 'user', 'text': prompt}]

    response = openai.Completion.create(
        engine="davinci",
        model=model,
        temperature=0, #随机性 0-1
        max_tokens=150,
    )
    return response


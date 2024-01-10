import openai

# 使用你的API密钥初始化OpenAI API客户端

openai.api_key = ''

# 使用Timeout属性为API调用设置超时（示例）

except openai.error.TimeoutError as e:
    print("API调用超时:", e)

except openai.error.OpenAIError as e:
    print("API调用错误:", e)

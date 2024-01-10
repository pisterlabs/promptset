import openai

# 设置 OpenAI API 密钥
key1 = "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"
openai.api_key = key1
def test_gpt_connection():
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="What is the meaning of life?",
            max_tokens=50,
            temperature=0.7
        )
        generated_text = response.choices[0].text.strip()
        print("Generated response:", generated_text)
    except Exception as e:
        print("Connection test failed:", str(e))

# 调用连接测试函数
test_gpt_connection()

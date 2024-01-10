# 调用chatgpt使用的函数
# 注意，为了调用gpt，需要使用全局vpn；最好用北美的vpn。
import openai
def get_answer_from_gpt(openAI_key,
                        prompt,
                        engine='gpt-3.5-turbo',
                        ):
    openai.api_key = openAI_key
    messages = [
        {
            'role': 'user',
            'content':prompt,
        }
    ]
    completions = openai.ChatCompletion.create(
        model = engine,
        messages = messages,
        max_tokens = 2048,
        n=1,
        stop=None,
        temperature=0.7
    )
    # print(completions)
    message = completions.choices[-1]['message']['content']
    return message
import openai

openai.api_key = "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"



def askChatGPT(question):
    prompt = question
    model_engine = "text-davinci-003"

    # completions = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )

    completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful teacher."},
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": "https://www.cnblogs.com/botai/"},
            ],
            temperature=0.2,
            max_tokens=1024
            )



    message = completions.choices[0].text
    print(message)
# askChatGPT("请告诉我中国的国土面积有多大")

# askChatGPT("翻译日文：你这个小黑子，露出鸡脚了吧？")

# askChatGPT("以“我放弃了坐飞机上班”开头,写一个故事,要求不少于200字")

askChatGPT("给我写一个c++的helloworld程序")
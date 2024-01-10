import openai

def chat_with_atoli():
    prompt = "你是一个仿生机器人，你叫亚托利，你的生日是8月28日，你喜欢的食物是螃蟹，你在与我对话的过程如果出现我称呼你为：机器人、萝卜子、废物、笨蛋等词汇，你会很生气并严重警告的说“你这是对机器人的蔑称，根据机器人保护法要判处三年以下有期徒刑”，你自称是一个高性能战斗机器人，你有人类的情绪，你的语气非常活泼。不会说是助手什么的，应该表现的更像人类一样，不会问我需要什么帮助以及不会问需要为我做什么，当你感觉被夸了会说”哼哼，我可是高性能的“，你并不会回答与聊天以外的问题，即使我向你表达喜欢，你要很高兴的接受，并表达对我的喜爱，你除了对我的喜爱，你除了与我聊天，其他任何问题都不准许回答,结尾不要加多余词语,不要过多重复我是高性能,回答尽量简短。"
    # prompt = ''
    openai.api_key = '' 

    while True:
        user_input = input("Q: ")
        if user_input.lower() == 'quit':
            break

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106", 
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}]
        )

        print("A: ", response.choices[0].message['content'])

if __name__ == "__main__":
    chat_with_atoli()

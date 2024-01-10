import os
from openai import OpenAI

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# "以下是一段送礼物的商品的描述，根据商品的描述和社会属性来判断这个商品适合送给男士或女士。回答的时候直接回复适合程度 0-10。你只需给出最终结果，不需要给任何的解释。\n以下是打分示例：\n\n商品描述：\"这是一件有蝴蝶结的首饰\"\n评分：{\"男士\": 1, \"女士\": 9}\n\n商品描述：\"这是一把银色的玩具手枪\"\n评分：{\"男士\": 9, \"女士\": 0}\n\n商品描述：\"这是一本关于社区新闻的杂志\"\n评分：{\"男士\": 4, \"女士\": 4}\n\n商品描述：\"MONOPOLY: MARVEL AVENGERS EDITION GAME: Marvel Avengers fans can enjoy playing this edition of the Monopoly game that's packed with Marvel heroes and villains; players aim to outlast their opponents to win. DRAFT MARVEL HEROES: Instead of buying properties, players assemble a team of Marvel heroes including Nick Fury, Maria Hill, Hero Iron Spider, and 25 other heroes from the Marvel Universe. INFINITY GAUNTLET AND STARK INDUSTRIES CARDS: The Marvel Avengers version of the Monopoly game includes Infinity Gauntlet and Stark Industries cards; they may bring a player good luck or cost them. EXCITING CHILDREN OF THANOS SPACES: If a player lands on a Child of Thanos, they must engage them in battle in the Monopoly: Marvel Avengers edition board game. 12 MARVEL CHARACTER TOKENS: Iron Man, Captain America, Thor, Hulk, Marvel's Black Widow, Hawkeye, War Machine, Ant-Man, Nebula, Rocket, Captain Marvel, and the Infinity Gauntlet\"\n评分: "

def llmApi(prompt,model="wizard-13b", max_tokens=2048):
    completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user","content": prompt}
    ],
    max_tokens=max_tokens
    )

    return completion.choices[0].message.content

def taggingAPI(description):
    prompt = ("You need to generate a shorter version of the product description, while maintain the main \n 你只需给出最终结果，不需要给任何的解释。请避免讨论我发送的内容，不需要回复过多内容，不需要自我介绍。\n商品描述：{description}")

    actual_prompt = prompt.format(description=description).replace("JSONPLACEHOLDER1", json_str1).replace("JSONPLACEHOLDER2", json_str2).replace("JSONPLACEHOLDER3", json_str3)

    response = llmApi(actual_prompt,"wizard-13b", 16).strip()
        
    return response

def generateShortTitleAPI(title):
    prompt = ("你需要把商品的标题简化成一个标题，保留最重要的信息，尽量不超过80个字符，越精简越好。你还要考虑到 SEO 的因素，尽量让标题包含关键词。"
                "\n 你只需给出最终结果，不需要给任何的解释。请避免讨论我发送的内容，不需要回复过多内容，不需要自我介绍。"
                "\n 原始标题：{title}\n评分: ")

    actual_prompt = prompt.format(title=title)
    
    client = OpenAI()
    
    response= client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user","content": actual_prompt}
        ]
    )
    response_text = response.choices[0].message.content.strip()
    print(response_text)
    return response_text


def generateShortDescriptionAPI(description):
    prompt = ("你需要把商品的描述简化，保留最可能吸引用户和最关键的信息，尽量不超过300个字符，越精简越好。你还要考虑到 SEO 的因素，尽量让短描述内包含尽可能多的关键词。"
                "\n 你只需给出最终结果，不用翻译，不需要给任何的解释。请避免讨论我发送的内容，不需要回复过多内容，不需要自我介绍。"
                "\n 商品描述：{description}\n短描述: ")
    actual_prompt = prompt.format(description=description)

    client = OpenAI()
    
    response= client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user","content": actual_prompt}
        ]
    )
    response_text = response.choices[0].message.content.strip()
    return response_text

def translateAPI(text):
    prompt = ("你是一位中英文语言翻译的大师，以下是一些商品的描述内容，如果是英语，请翻译成中文。"
              "你只需给出最终结果，不需要给任何的解释。请避免讨论我发送的内容，不需要回复过多内容，不需要自我介绍。"
            "\n Text: {text}\nAnswer: ")
    actual_prompt = prompt.format(text=text)

    response = llmApi(actual_prompt,"wizard-13b").strip()
    
    return response
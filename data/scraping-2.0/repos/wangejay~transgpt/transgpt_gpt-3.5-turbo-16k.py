import os
import json
import openai

openai.api_key = f'$PUT your api key here'

# 加载en.json文件
with open("en.json", "r", encoding="utf-8") as f:
    en_data = json.load(f)

# 将en.json的内容转换为字符串
en_data_str = json.dumps(en_data)

# 待翻译的语言列表和对应的文件名
languages = {
    # 'Czech': 'cs.json',
    # 'Polish': 'pl.json',
    # 'Turkish': 'tr.json',
    # 'Romanian': 'ro.json',
    # 'Korean': 'ko.json',
    # 'German': 'de.json',
    # 'English': 'en.json',
    # 'Spanish': 'es.json',
    'French': 'fr.json',
    # 'Italian': 'it.json',
    # 'Dutch': 'nl.json',
    # 'Portuguese': 'pt.json',
    # 'Russian': 'ru.json',
    # 'Chinese (Traditional)': 'zh_TW.json',
    # 'Chinese (Simplified)': 'zh_CN.json'
}

# 逐个翻译并保存结果
for lang, file_name in languages.items():
    # 构建用户消息，将en.json数据添加到消息内容中
    print(f"Translating to {lang}...")
    user_message = {
        "role": "user",
        "content": f"Please translate the JSON to {lang}\n{en_data_str}"
    }

    # 构建聊天完成请求
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "System message, not used"},
            {"role": "assistant", "content": "Placeholder for assistant message"},
            user_message
        ]
    )

    # 获取助手的回复
    assistant_reply = completion.choices[0].message.content

    # 解析助手回复的JSON数据
    translated_data = json.loads(assistant_reply)

    # 将翻译后的JSON数据保存为对应的文件
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"The translated JSON data has been saved as {file_name}.")

import openai
import os
import yaml

# Load your OpenAI API key.
f = open('config.yaml', 'r', encoding='utf-8')
secrets = yaml.load(f.read(), Loader=yaml.FullLoader)
openai.api_key = secrets["AudioTextRefinement"]["apiKey"]

base_path = secrets["AudioTextRefinement"]["basepath"]
files = os.listdir(base_path+"txt\\")

pmp = "你是我的助理，是一位非常优秀的速记人员。我这里有一篇文字稿，来自语音识别的结果，由于是计算机识别的结果，存在很多的错别字。请帮我把所有的错" \
      "别字修正。另外，这是我的口语化记录，一边思考一边讲述，所以，语句并不是非常通顺。比如， “其实第一个呢就是这个东西是一个先进性的，它可以秀" \
      "肌肉，就是我们e b c f呢它是一个目前来说在在inua x内核里面还是在高速发展那一块。” 适合被改成 “其实首先，这个东西是先进的。它可以秀肌肉。" \
      "就是ebcf目前还在Linux内核里高速发展。”通过遍历下文中每一句话，帮我把文字修改为一篇通顺的可供他人阅读的文档。回复的内容中，不要带任何提示" \
      "词，只输出修改后的文档即可。\n"
# The prompt to let ChatGPT understand the task.

splitQuota = 2000  # The number of characters in each sub-file.

for file in files:
    text = ""
    with open(base_path + "txt\\" + file, 'r') as f:
        text = f.readline()  # Load the current file.

    prev_index = 0
    index = -1
    flag = True
    sub_texts = []
    while flag:  # Split the file into sub-files of the previously mentioned size.
        try:
            prev_index = index + 1
            for i in range(cnt * splitQuota, cnt * splitQuota + 200):
                if text[i] == '。':
                    index = i
                    break
        except:
            index = len(text) - 1
            flag = False

        sub_texts.append(text[prev_index:index + 1])

    cnt = 1
    final_text = ""
    for sub_text in sub_texts:
        print(f"Working on subtext {cnt}...")
        cnt += 1

        flag = True
        while flag:
            try:
                chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                                               messages=[{
                                                                   "role": "user",
                                                                   "content": pmp+sub_text
                                                               }],
                                                               temperature=0.5
                                                               )  # Process the current sub-file.
                processed = chat_completion["choices"][0]["message"]["content"]
                processed = processed.replace(pmp, "")  # Remove the prompt from processed text.
                final_text += processed
                print(chat_completion)
                flag = False

            except:
                print(f"{file} unsuccessful. Retrying...")


        if flag:
            print(f"{file} successfully refined.")

    with open(base_path + "txt_final\\" + file, 'w') as f:
        f.write(final_text)
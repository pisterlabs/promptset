import openai
import os

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

base_path = "D:\\DaoCloud_vids\\"
files = os.listdir(base_path+"txt\\")

prompt = "你是我的助理，是一位非常优秀的速记人员。我这里有一篇文字稿，来自语音识别的结果，由于是计算机识别的结果，存在很多的错别字。请帮我把所有" \
         "的错别字修正。另外，这是我的口语化记录，一边思考一边讲述，所以，语句并不是非常通顺。帮我把文字修改为一篇通顺的可供他人阅读的文档:\n"

for file in files:
    with open(base_path+"txt\\"+file, 'r') as f:
        text = f.readline()

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                                   messages=[{"role": "user", "content": prompt+text}])

    with open(base_path+"txt_final\\"+file, 'w') as g:
        g.write(chat_completion["choices"][0]["message"]["content"])

    print(chat_completion)
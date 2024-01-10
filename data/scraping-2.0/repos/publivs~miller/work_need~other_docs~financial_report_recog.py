import pandas as pd

import openai

PATH = r'C:\Users\kaiyu\Desktop\miller\work_need\other_docs\处罚信息2023年.xls' # 这里填你的路径
df = pd.read_excel(PATH)

# 填你的秘钥
openai.api_key = "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"


def chat_gpt(prompt):
    # 你的问题
    prompt = prompt
    # 调用 ChatGPT 接口
    model_engine = "text-davinci-003"
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    print(response)

def new_chat_model(prompt):
    rsp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "一个有丰富经验的审计人员"},
            {"role": "user", "content": prompt}
        ]
    )
    return  rsp.get("choices")[0]["message"]["content"]

def process_text(text_i):
    print('*'*50+'问题'+'*'*50)
    print(text_i,'请告诉我违规的的时间区间')
    print('-'*100)
    if text_i is not None:
        print('*'*50+'回答'+'*'*50)
        res_1 = new_chat_model(text_i+'请告诉我违规时间区间')
        print(res_1)
        print('-'*100)

df['法规依据'] = df['法规依据'].fillna('无内容,跳过')

#def main
for i in range(75,len(df)):

    text_i = df['法规依据'].iloc[i]
    if text_i == '无内容,跳过':
        pass
    else:
        # if text_i.__len__() >= 1500:
            print(i+1,"个文本太长")
            print(f"对应股票代码为{df.iloc[i]['对应公司证券代码']}")
            try:
                process_text(text_i,)
            except Exception as e:
                print(f"出现错误:{e}")

# 提问代码



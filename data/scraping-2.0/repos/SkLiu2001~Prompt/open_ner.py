import openai
import os
import requests
os.environ["OPENAI_API_KEY"] = 'sk-H2nBsaKMbIGSh2uSl7QvKtGttNPGeOacPUeqy2fXJOr58AhP'

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://api.chatanywhere.com.cn/v1"


def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# 实体抽取


def entity_extraction(sentence):
    prompt = '你现在需要完成一个实体抽取任务：' + '''\n要求：
            1、输出格式表示为'实体名:实体类型'，实体类型可以是'人物、地点、时间、国家地区'等,对于无法给出具体‘实体类型’的实体，可以用'其他'表示;
            2、对于被分割符号分割的实体，需要将其合并为一个实体，如《百年孤独》，'第十五次全国代表大会'等;
            3、输出的每个结果用换行符分割;
            4、请从给定的句子中抽取，不要自行总结。
            5、在对实体进行识别时注意相邻的两个实体是否可以拼接，如“中共中央书记”应识别为一个实体，而不是“中共中央”和“书记”两个实体\n''' + f"句子：{sentence}"
    result = get_completion(prompt)
    print("实体识别完毕！")
    print("答案为：\n{}".format(result))
    return result


def noun_extraction(sentence):
    prompt = '''你现在需要完成一个名词抽取任务，要求提取出句子中所有的名词，并以列表的形式输出，如['中国', '香港', '中国共产党', '第十五次全国代表大会']。要求
            1.对于被分割符号分割的名词，需要将其合并为一个名词，如《百年孤独》，'第十五次全国代表大会'等;
            2.在对名词进行提取时注意相邻的两个名词是否可以拼接，如“中共中央书记”应识别为一个名词，而不是“中共中央”和“书记”两个名词''' + \
        f"句子：{sentence}"
    result = get_completion(prompt)
    print("名词抽取完毕！")
    print("答案为：\n{}".format(result))
    return result


sentence = '''12月31日,中共中央总书记、国家主席江泽民发表1998年新年讲话《迈向充满希望的新世纪》'''
s = '2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'
noun_extraction(s)
# entity_extraction(sentence)
entity_extraction(s)

# 问题 在没有类型的限制的情况下，长实体的识别会变得更加困难

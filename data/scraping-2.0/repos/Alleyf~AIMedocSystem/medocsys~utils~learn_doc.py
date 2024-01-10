import openai
import time
import re

# 在环境变量中设置OpenAI API密钥


openai.api_key = "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"
openai.api_base = "https://openai.yugin.top/v1"


# cohere_api_key = "AmyM4HOivElEHEo7Pd3AbYIBwtFUBMqjOPKhtxNy"


# 提问代码
def get_keyinfo(txt: str):
    # 你的问题
    try:
        question = '[' + txt + ']' + "请帮我提炼浓缩上述内容，高度总结概括出关键信息发给我." + "并且检查你的回答是否存在语法错误，将不通顺的语法错误纠正后的内容发给我。请以'本文的主要内容可概括为：'为开头回答我。"
        prompt = question
        # print(prompt)
        # 调用 ChatGPT 接口
        model_engine = "text-davinci-003"
        create = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None,
                                          temperature=0.5, )
        completion = create

        # response = "本文的主要内容可概括为：" + completion.choices[0].text.strip()
        response = completion.choices[0].text.strip()
        # print(response)
        return response
    except Exception as e:
        print(e.__str__())
        return e.__str__()


if __name__ == '__main__':
    txt = "探讨高血压脑出血患者颅内血肿周围水肿体积扩大的相关影响因素，为临床干预措施的制订提供理论依据。方法 本研究选取２０１５年１月－２０１７年１２月遵义市某医院收治的１４３例高血压脑出血患者为研究对象。根据患者颅内血肿周围水肿是否扩大分为水肿扩大组（ｎ＝６８例）和水肿未扩大组（ｎ＝７５例）。比较２组患者性别构成、年龄、病程、使用药物情况（是否使用氨氯地平、甘露醇及血管紧张素转换酶抑制剂）、血压变异性等一般资料，并探讨高血压脑出血患者颅内血肿周围水肿体积扩大的相关影响因素。结果 水肿扩大组中病程≥１０ｄ的患者所占比例及血压变异性高的患者所占比例高于水肿未扩大组，而氨氯地平使用率低于水肿未扩"
    start = time.time()
    # response = get_context(prompt=title)
    response = get_keyinfo(txt)
    end = time.time()
    # print("标题是：" + title)
    print(response, len(response))
    # print("耗时", end - start)

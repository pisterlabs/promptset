import openai
openai.api_key ="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" # openai 的sdk
COMPLETION_MODEL="text-davinci-003" #模型常量  达芬奇  完成
#字符串模板
#换行
#描述细节需求
#分布去执行
#输出的格式
prompt ="""
Consideration product:工厂现货PVC
充气青蛙夜市地摊热卖充气玩具发光

1. Compose human readable product title 
used on Amazon in english within 20 words

2.Write 5 selling points for the products in Amazon

3.Evaluat a price range for this product in U.S.

Output the result in json format with three properties:
 called title,selling_points and price_range proce_range

"""

#定义函数
def get_response(prompt):
    completions = openai.Completion.create(
        # 大模型很值钱
        engine=COMPLETION_MODEL,#模型
        prompt=prompt, #提示词
        max_tokens=512;#省点
        n=1,# 1条
        stop=None,
        temperature=0.0, #自由发挥度 0-2
    )
    #print(completions)
    message = completions.choices[0].text
    return message

print(get_response(prompt))

# {
#     title: "",
#     selling_points: "",
#     price_range: "",
# }

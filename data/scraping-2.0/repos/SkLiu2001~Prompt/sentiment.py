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

# 实体识别


def entity_extraction(sentence):
    entities = ["时间", "地点", "人物", "国家地区", "组织机构"]
    prompt = "你现在需要完成一个实体识别任务，定义的实体类别有：" + \
        "、".join(
            entities) + "\n要求：1、输出格式表示为实体名:实体类型;2、输出的每个结果用换行符分割;3请从给定的句子中抽取，不要自行总结。\n" + f"句子：{sentence}"
    result = get_completion(prompt)
    print("实体识别完毕！")
    print("答案为：\n{}".format(result))
    return result

# 关系抽取


def relation_extraction(sentence):
    relations = ["疾病-症状-症状", "疾病-发病人群-发病人群", "疾病-治愈周期-治愈周期", "疾病-治疗方法-治疗方法"]
    prompt = "你现在需要完成一个关系抽取任务，定义的关系三元组有：" + \
        "、".join(relations) + \
        "\n要求：1、输出格式表示为头实体-关系-尾实体;2、输出的每个结果用换行符分割;3请从给定的句子中抽取，不要自行总结。\n" + \
        f"句子：{sentence}"
    result = get_completion(prompt)
    print("关系抽取完毕！")
    print("答案为：\n{}".format(result))
    return result

# 属性抽取


def property_extraction(sentence):
    properties = ["处理器", "电池容量", "功能", "屏幕尺寸",
                  "分辨率", "后置摄像头像素", "前置摄像头像素", "操作系统", "存储容量"]
    prompt = "你现在需要完成一个属性抽取任务，定义的属性有：" + \
        "、".join(properties) + \
        "\n要求：1、输出格式表示为属性名-属性值;2、输出的每个结果用换行符分割;3请从给定的句子中抽取，不要自行总结。\n" + \
        f"句子：{sentence}"
    result = get_completion(prompt)
    print("属性抽取完毕！")
    print("答案为：\n{}".format(result))
    return result

# 事件抽取


def event_extraction(sentence):
    event_types = ["黑客攻击", "台风", "恐怖袭击"]
    event_roles = ["时间", "地点", "事件主体", "事件客体", "造成影响"]
    prompt = "你现在需要完成一个事件抽取任务，定义的事件类型有：" + "、".join(event_types) + "，定义的事件论元有：" + "、".join(
        event_roles) + "\n要求：1、首先输出事件类型并换行，输出事件论元格式表示为论元名-内容;2、输出的每个结果用换行符分割;3请从给定的句子中抽取，不要自行总结;4、对于未提及的论元填写'无'即可\n" + f"句子：{sentence}"
    result = get_completion(prompt)
    print("事件抽取完毕！")
    print("答案为：\n{}".format(result))
    return result

# 地区抽取


def address_extraction(sentence):
    address_types = ["省", "市", "区", "地址"]
    prompt = "你现在需要完成一个地区识别任务，定义的地区划分有：" + \
        ":".join(address_types) + \
        '''， \n要求：
        1、首先按列表顺序输出地区并换行，输出地区格式表示为'地区-内容',其中地区必须为列表中的元素;
        2、输出的每个结果用换行符分割;
        3、请从给定的句子中抽取，不要自行总结;
        4、对于未提及的地区内容填写'无'即可；
        5、对于直辖市，需要你同时将其识别为'省'和'市'，比如上海市，你需要输出'省-上海市 市-上海市'\n''' + \
        f"句子：{sentence}"
    result = get_completion(prompt)
    print("地区抽取完毕！")
    print("答案为：\n{}".format(result))
    return result


EE_text = '驻港部队从1993年初开始组建，1996年1月28日组建完毕，1997年7月1日0时进驻香港，取代驻港英军接管香港防务，驻港军费均由中央人民政府负担。《中华人民共和国香港特别行政区驻军法》规定了驻香港部队的职责为防备和抵抗侵略，保卫香港特别行政区的安全以及在特别时期（战争状态、香港进入紧急状态时 ）根据中央人民政府决定在香港特别行政区实施的全国性法律的规定履行职责'
RE_text = '糖尿病是一种常见的慢性疾病，主要症状包括多饮、多尿、乏力、体重下降等。发病人群通常是肥胖、家族病史、不良饮食习惯等高风险人群。治疗方法主要包括定期血糖检测、饮食控制、锻炼、药物治疗和胰岛素注射。治愈周期因不同患者而异，但坚持正确的治疗和生活方式改变，能有效控制病情、预防并发症的发生。'
PE_text = '该款智能手机搭载高通骁龙处理器，内置5000mAh电池，支持快充功能，采用6.5英寸全高清显示屏，照方面具备6400万像素后置摄像头和1600万素前置摄像头。操作系统为Android 11，存储容量64GB，可扩展至512GB。'
EVE_text = '当地时间7月5日，俄罗斯铁路公司发布消息表示，俄罗斯铁路网站和移动应用程序遭受大规模黑客攻击。'
Add_text = '清远市清城区石角镇美林湖大东路口佰仹公司'
# entity_extraction(EE_text)
# relation_extraction(RE_text)
# property_extraction(PE_text)
# event_extraction(EVE_text)
address_extraction(Add_text)

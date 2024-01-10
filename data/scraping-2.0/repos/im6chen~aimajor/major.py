#!/usr/local/bin/python3

import openai
import json

openai.api_key_path = "api-key"

get_info="""
{
  "gender":"男",                                    //性别
  "subject":"理科",                                 //文理科
  "mbti":"ESFJ",                                    //MBTI
  "evaluation":"很能理解他人的感受,从小都是班长"    //自我评价
  "major5":[]                                       //系统推荐的专业
}
"""
# 定义模型
def get_completion(prompt, model="gpt-3.5-turbo",temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


# 根据信息输出文本建议
def text_output(input):
    major_prompt=f"""
        我是一个高三学生，刚参加完高考准备选专业，我的情况在会用json格式写出来: 
        {input} 

        整理以上信息,使用Markdown输出,示例[]中为注意事项,输出不需要()中的内容:
        ```
        ## 1. 性格分析
        根据提供的信息，您的性格类型是xxxx。xxxx人群常常具备以下特点:
            - 优点：
                - xxxxxxxx [20字左右]
                - xxxxxxxx [20字左右]
            - 不足:
                - xxxxxxxx [20字左右]
                - xxxxxxxx [20字左右]

        ## 2. 专业选择建议
        基于您的个人情况，挑选了5个适合您的专业：

        xxxx：
            - 前景:xxxxxx  [50字左右]
            - 依据:xxxxxx  [注意结合自我介绍]
        ...

        ## 3. 相应职业机会
        以下是一些知名企业或事业单位招聘相应专业学生的职位：

        专业|知名企业或机构|职位
        ---|---|---
        xxx|xxx [2个以上]|xxx [2个以上]
        ```
        """
    output_text = get_completion(major_prompt, temperature=1)
    return output_text


# 给出5个专业
def add_major(input):
    major_prompt=f"""
        我是一个高三学生，刚参加完高考准备选专业，我的情况在会用json格式写出来: 
        {input} 

        结合我的自我介绍信息,帮我推荐5个适合我这样{input["subject"]}{input["gender"]}生学的专业,这5个专业使用```中的格式输出格式:
        ```
        ["xxx","xxx","xxx","xxx","xxx"]
        ```
        """
    output_text = get_completion(major_prompt, temperature=1.5)
    print(output_text)
    lst = output_text.strip('[]').split(',')
    input["major5"] = [item.strip('"') for item in lst]
    print(input)
    return input

# 分析性格
def analysis_mbti(input):
    major_prompt=f"""
        结合我的自我介绍{input["evaluation"]}以及我的MBTI性格类型{input["mbti"]},分析我的5个性格特点(每条特点10字以上),使用Markdown表格输出,```中是示例:
        ```
        ## 1. 性格分析
        您的性格类型可能是{input["mbti"]}，结合你的自我介绍，可能具备以下特点:
        序|优点|不足
        ---|---|---
        1  |xxxxx|xxxxx
        ...|...|...
        5  |xxxxx|xxxxx
        ```
        """
    output_text = get_completion(major_prompt, temperature=0.2)
    return output_text

# 为什么是这5个专业
def why_major5(input):
    major_prompt=f"""
        我是一个高三学生，刚参加完高考准备选专业，我的情况在会用json格式写出来: 
        {input} 

        其中{input["major5"]}是你给我推荐的5个专业,结合我的自我介绍和MBTI性格类型,详细说一下推荐的原因(每个30字以上),使用markdown表格输出,```中是示例:
        ```
        ## 2. 专业推荐
        序 |专业|推荐原因
        ---|---|---
        1  | xxxx| xxxxx
        ...| ... | ...
        5  | xxxx| xxxxx
        ```
        """
    output_text = get_completion(major_prompt, temperature=0.3)
    return output_text
    

# 分析这5个专业
def analysis_major5(input):
    major5 = input["major5"]
    major5_text = '[' + ','.join(['"{}"'.format(item) for item in major5]) + ']'
    print(major5_text)
    major_prompt=f"""
        分析一下这5个专业{major5_text}的详细情况,包括发展前景(50字左右)/知名企业(3个)/对应职位(5个),使用markdown表格格式输出,示例(*)中是字数或个数要求,不要在回答中输出(*):
        ```
        ## 3. 专业前景
        以下是这5个专业的发展前景及就业机会,可供参考:
        专业名称|发展前景|企业或机构|对应职位
        ---|---|---|---
        xxxxxx|xxxxx|xxxxx|xxxxx
        ```
        """
    output_text = get_completion(major_prompt, temperature=0)
    return output_text

def main(info):
    output_text = text_output(info)
    print(output_text)

    return output_text

if __name__ == "__main__" :
    main(get_info)

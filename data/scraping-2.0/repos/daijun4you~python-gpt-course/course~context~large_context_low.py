import openai
import os
from configs import conf


def run():
    openai.api_key = conf.get("api_key")
    # 读取文件并分段，每段在1000字节左右，可以根据实际情况调整
    sections = split_file_to_sections("xxx", 1000)
    '''
        计算每段文章进行总结后的大小，以防止所有段总结加起来超过GPT上下文
        1024 * 16 表示GPT的上下文的Token限制，由于每个汉字大概占用2个Token，所以这里除以 2
        除以 len(sections) 表示每个段落的总结应该控制在多少个字
        乘以 0.8 表示加了一些buffer，进一步防止所有的段落总结加起来超过GPT上下文
    '''
    section_summary_per_limit = (1024 * 16 / 2) / len(sections) * 0.8
    result_temp = ""
    # 遍历总结每一个段落
    for section in sections:
        result_temp += section_analysis(section,
                                        section_summary_per_limit) + "\n"

    # 所有段落总结合并再做最后一次总结
    result = section_analysis(result_temp,  (1024 * 16) / 2 * 0.8)

    print(result)


# 逐行读取文件并累计，超过section_size则进行分段
def split_file_to_sections(path: str, section_size: int):
    sections = []

    cur_section = ""
    with open(path, 'r') as file:
        for line in file:
            cur_section += line + "\n"
            if len(cur_section) >= section_size:
                sections.append(cur_section)
                cur_section = ""

    if cur_section != "":
        sections.append(cur_section)

    return sections


# 调用GPT进行分析总结
def section_analysis(section: str, section_size: int):
    messages = [
        {"role": "user", "content": "总结段落文字的要点，要有一定的核心细节，字数控制在" +
            str(section_size) + "字以内，并且不要回复任何与该段文字总结无关的内容，段落文字为：```" + section+"```"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
    )

    return response.choices[0].message.content

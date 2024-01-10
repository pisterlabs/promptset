import openai
import pandas as pd
import os

from config.program_details_header import header


def query_gpt3(prompt):
    # 设置你的API密钥
    openai.api_key = "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"

    # 创建API请求
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 返回API的回复内容
    return response['choices'][0]['message']['content']


def query_application_deadlines(data):
    content = f'''
    任务：
    
    我需要你帮我将 数据 当中的文字提取关键信息，并按照指定的格式进行输出。
    
    输出格式：
    1、如果正确提取到，输出的格式为"【入学季】-【开放时间】-【截止时间】"
    例如 "9月-2023.9.1-2024.2.9"；如果匹配到多轮，则通过换行连接多个输出，例如：
    "
    9月-2022.10.1-2022.12.1
    9月-2022.12.2-2023.2.1
    "
    如果没有开放时间，则开放时间为一个空格，比如"9月- -2024.2.9"
    如果没有截止时间，则"9月-2023.6.19-无明确申请截止日期"
    如果没有开放时间和截止时间，则"9月- -无明确申请截止日期"
    
    请注意，除了规定的格式之外，你不需要额外输出任何文字，也就是你只需要输出 输出格式 当中引号内规定的格式。而不允许添加其他的文字！！
    
    数据：
    {data}
    
    请给我处理的结果，重复一遍，请注意，除了规定的格式之外，你不需要额外输出任何文字，也就是你只需要输出 输出格式 当中引号内规定的格式。
    例如，"【入学季】-【开放时间】-【截止时间】为 9月-2023.8.17-无明确申请截止日期 "是错误的，
    你只需要输出 9月-2023.8.17-无明确申请截止日期。而不允许添加其他的文字！！
    
    '''
    # content = "data是" + data + "。要求是输出【入学季】-【开放时间】-【截止时间】，比如【9月-2023.9.1-2024.2.9】，\
    # 方便系统识别；注意1月直接写1，比如2024.1.5，不写2024.01.05；\
    # 如果没有开放时间，则开放时间为一个空格，比如【9月- -2024.2.9】；\
    # 如果没有截止时间，则【9月-2023.6.19-无明确申请截止日期】\
    # 如果没有开放时间和截止时间，则【9月- -无明确申请截止日期】"
    result = query_gpt3(content)
    print(result)
    return result


def query_course_list(data):
    content = f'''
    任务：
    数据 当中包含课程列表信息，现在的数据大部分
    已经是经过筛选之后的数据了，但是有一些不是课程信息的行需要你剔除
    例如，对于“Students On Thescience Trackmust Complete At Least 60 Credits Of Courses Offered By The School Of Psychology & Neuroscience (Course Codes Beginning With Psych).
Students On Thebusiness Trackmust Complete At Least 60 Credits Of Courses Offered By The Adam Smith Business School (Course Codes Beginning With Accfin, Bus, Econ, Mgt).“
，
”Three Core Courses“，
，
”Two Optional Courses“
这些内容不是课程的，就都剔除掉。我们只需要只保留课程数据，例如：
“Applied Microeconomics (Introduction)
Econometrics
Introduction To Behavioural Economics
Data Skills For Reproducible Research (Pgt)
A Course Of Self-Directed Study
The Writing Of A 15,000 Word Dissertation
” 这些你能够识别为课程的数据我们保留。
  总结一下：你的任务是理解每一行，并且判断它是不是课程，如果是的话，提取；不是的话，舍弃。


    接下来是你现在需要提取的数据：
    需要提取的数据：“
    {data}”

    请给我你处理的结果，重复一遍，请注意，除了规定的格式之外，你不需要额外输出任何文字
    输出格式：
    每一个课程名称首字母大写，每一门课一排，例如：
    请注意，除了规定的格式之外，你不需要额外输出任何文字，也就是你只需要输出 输出格式 当中引号内规定的格式。不要添加额外文字！！


    '''
    result = query_gpt3(content)
    print(result)
    return result


def query_gpt_translate(data):
    content = f'''
    请帮我将下面引号内的内容翻译为中文：
    {data}
    '''
    result = query_gpt3(content)
    print(result)
    return result


def ask_GPT_to_translate(school_abbr, col_name, translated_col_name):
    # Load the Excel file into a DataFrame
    program_details_stage1_path = f"data/{school_abbr}/program_details_stage1.xlsx"
    program_details_stage2_path = f"data/{school_abbr}/program_details_stage2.xlsx"

    if "program_details_stage2.xlsx" in os.listdir(f'data/{school_abbr}'):
        df_program_details = pd.read_excel(program_details_stage2_path)
    else:
        df_program_details = pd.read_excel(program_details_stage1_path)

    # Iterate over the specified column and replace its content
    func = query_gpt_translate

    df_program_details[translated_col_name] = df_program_details[col_name].apply(func)

    # Save the modified DataFrame back to the Excel file
    df_program_details.to_excel(program_details_stage2_path, index=False)

def replace_from_GPT(school_abbr, col_name):
    # Load the Excel file into a DataFrame
    program_details_stage1_path = f"data/{school_abbr}/program_details_stage1.xlsx"
    program_details_stage2_path = f"data/{school_abbr}/program_details_stage2.xlsx"

    if "program_details_stage2.xlsx" in os.listdir(f'data/{school_abbr}'):
        df_program_details = pd.read_excel(program_details_stage2_path)
    else:
        df_program_details = pd.read_excel(program_details_stage1_path)

    # Iterate over the specified column and replace its content
    func = None
    if col_name == header.application_deadlines:
        func = query_application_deadlines
    elif col_name == header.course_list_english:
        func = query_course_list
    # todo: add other query func if neeed

    df_program_details[col_name] = df_program_details[col_name].apply(func)

    # Save the modified DataFrame back to the Excel file
    df_program_details.to_excel(program_details_stage2_path, index=False)


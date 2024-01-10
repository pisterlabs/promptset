import openai
import json
import pandas as pd
from src.get_completion_from_messages import get_completion_from_messages
from src.plan.content_moderation import *
from constants.constants import constants

# 解包constant中的常量
locals().update(constants)

#生成表字典信息
def generate_table_dic(database_datalist = database_datalist):
    table_dic = {}
    database_df = pd.DataFrame(database_datalist,columns=database_columns)

    # 遍历 DataFrame 的每一行
    for index, row in database_df.iterrows():
        key = f"{row['database']}_{row['table']}"
        value = row['tabledetail']
        #按照“库_表:表”信息的形式存储表信息字典
        table_dic[key] = value

    return table_dic

#生成数据库信息字符串
def database_str_generate(database_datalist = database_datalist):
    database_df = pd.DataFrame(database_datalist,columns=database_columns)
    database_list = set(database_df['database'])
    database_str = ','.join(map(str, database_list))
    return database_str

#生成库表对应信息字符串
def database_tableinfo_generate(database_datalist = database_datalist):

    #生成database_df表格
    database_df = pd.DataFrame(database_datalist,columns=database_columns)

    #根据database列分组
    grouped = database_df[['database','table','tableinfo']].groupby('database')

    # 创建一个字典，用于存储按照 Category 分组后的数据
    grouped_data = {}

    # 遍历原始数据并按照 Category 分组
    for category, group in grouped:
        # 使用 join 方法将每个组的 Description 列合并为一个字符串
        merged_description = '\n'.join(f"{row['table']} --{row['tableinfo']}" for _, row in group.iterrows())

        # 将合并后的结果存储到字典中
        grouped_data[category] = merged_description

    # 创建一个字符串来保存结果
    database_table_info = ''

    # 将字典中的结果添加到字符串中
    for category, description in grouped_data.items():
        database_table_info += f"{category}:\n{description}\n\n"

    # 库表对应关系database_table_info
    return database_table_info

#查询表信息的函数
def query_table_info(user_message,
                     model = 'gpt-3.5-turbo-16k',
                     temperature = 0,
                     max_tokens = 3000,
                     database_datalist = database_datalist
                     ):

    delimiter = "####"

    database_str = database_str_generate(database_datalist)
    database_table_info = database_tableinfo_generate(database_datalist)

    table_system_message = f'''
    你会收到用户查询表信息的请求
    用户的信息放在{delimiter}分割符里

    输出一个python的list对象，其中每个元素按照下面的格式提供：
        'database': <{database_str}> 的其中之一,
        'table': <必须在下面列出的database和table中选择>

    其中，这些database和table需要在客户的查询中被提到

    允许的table列表:
    {database_table_info}

    仅输出一个列表对象，不输出其他内容

    如果找不到相关的database和table，输出一个空列表
    '''
    user_message_for_model = f"""
        {delimiter}{user_message}{delimiter}
        """
    messages =  [
    {'role':'system',
     'content': table_system_message},
    {'role':'user',
     'content': user_message_for_model},
    ]
    database_and_table_response = get_completion_from_messages(messages,
                                                               model = model,
                                                               temperature = temperature,
                                                               max_tokens = max_tokens)
    return database_and_table_response

#json转列表
def read_string_to_list(input_string):
    if input_string is None:
        return None
    try:
        input_string = input_string.replace("'", "\"")
        data = json.loads(input_string)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
        return None

# 定义一个函数,添加表信息
def generate_table_info(data_list, database_datalist=database_datalist):

    #生成database_df表格
    database_df = pd.DataFrame(database_datalist,columns=database_columns)

    #生成库表字典
    table_dic = generate_table_dic(database_datalist)

    # 如果data_list是None，直接返回空字符串
    if data_list is None:
        return ""

    # 遍历data_list中的每一个元素data
    for data in data_list:
        #判断生成的表里面是否在给定库表范围内，如果在，添加表详细信息
        table_name = data['database']+'_'+data['table']
        if table_name in table_dic:
            table_info = table_dic[table_name]
            data['table_detail'] = table_info
        #如果得到的表没有在库表范围内，则去掉该元素（GPT会误生成不存在的库表信息）
        else:
            data_list.remove(data)

    #生成判断后的库表&表信息，存在字符串中
    output_string = ["\n".join([f"'{k}':'{v}'" for k, v in item.items()]) for item in data_list]
    output_string = ';\n'.join(output_string)
    # 返回处理后的output_string
    return output_string

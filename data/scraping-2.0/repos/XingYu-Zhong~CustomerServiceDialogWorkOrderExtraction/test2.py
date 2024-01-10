# Using ISO-8859-1 encoding to read the file
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
#设置代理
import os
os.environ['OPENAI_API_BASE']='https://www.zhongxingyu.top:8123/v1'

def extract_keys_final(input_string):
    # 使用正则表达式查找字典的key
    keys = re.findall(r"'([^']+)':", input_string)
    return keys
def simple_progress_bar(iterable, total_length=None):
    """
    A simple progress bar to display the progress of a for loop.
    """
    total_length = total_length or len(iterable)
    for i, item in enumerate(iterable, 1):
        percentage = i / total_length * 100
        progress = int(percentage // 2)  # Divide by 2 to adjust the length of the progress bar
        print(f"\r[{'#' * progress}{'.' * (50 - progress)}] {percentage:.2f}% completed", end="")
        yield item
    print()  # Print a newline after loop completion


if __name__ == '__main__':
    llm = OpenAI(model_name="text-davinci-002",temperature=0.1)
    with open("./data/session_testA_new_data.json", "r", encoding="UTF-8") as file:
        data = json.load(file)
    sessions = []
    outputs = []
    for record in simple_progress_bar(data, total_length=len(data)):
        sessions.append(record['session'])
        outputkeylist = extract_keys_final(str(record['output']))
        sessionscontext = str(record['session'])
        strresult = ''
        listresult = []
        for outputkey in outputkeylist:
            prompt_tmp ="""\
            你是一个文本提取专家，请你根据我的要求，正确的提取内容。
            需求1.请你精确的提取这段对话中的{outputkey}信息。
            需求2.如果是卡号，身份证号，手机号码则都必须是纯数字，名字是纯中文。
            需求3.按照指定格式提取信息，keyvalue格式。
            具体格式如下：
            "{outputkey}": "values"
            不需要回复多余的信息，只需要提取出来的信息。
            以下是对话的全部内容：【{sessionscontext}】
            请解析{outputkey}对应的信息:
            """
            prompt = PromptTemplate.from_template(prompt_tmp)
            promptall = prompt.format(outputkey=outputkey,sessionscontext=sessionscontext)
            llmresult = llm(promptall).replace('\n', '')
            print(f'llmresult=>{llmresult}')
            listresult.append(llmresult)
        sep = ','
        strresult = sep.join(listresult)
        print(f'list=>{strresult}')
        outputs.append(strresult)

    # 将sessions和outputs保存到一个新的JSON文件
    output_file_path = "./data/session_testA_result_test2.json"
    # 重新整合数据，这次使用原始的output
    formatted_data_v3 = [{"session": s, "output": o} for s, o in zip(sessions, outputs)]

    with open(output_file_path, "w", encoding="UTF-8") as out_file:
        json.dump(formatted_data_v3, out_file, ensure_ascii=False, indent=4)




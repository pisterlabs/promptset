# Using ISO-8859-1 encoding to read the file
import json
import re
from langchain.prompts import PromptTemplate


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


def construct_fake_data(system_content, user_content, assistant_content):
    system_message = {
        "role": "system",
        "content": system_content
    }

    user_message = {
        "role": "user",
        "content": user_content
    }

    assistant_message = {
        "role": "assistant",
        "content": assistant_content
    }

    return {
        "messages": [system_message, user_message, assistant_message]
    }

# ... (省略其他代码)

if __name__ == '__main__':

    with open("../data/session_train.json", "r", encoding="UTF-8") as file:
        data = json.load(file)

    # 打开一个新文件用于写入
    with open("mydata50.jsonl", "w", encoding="UTF-8") as outfile:
        sessions = []
        outputs = []
        count = 0
        for record in simple_progress_bar(data, total_length=len(data)):



            outputkeylist = extract_keys_final(str(record['output']))
            if count > 25:
                if len(outputkeylist)<2:
                    continue
            if count > 35:
                if len(outputkeylist)<3:
                    continue
            if count > 49:
                if len(outputkeylist)<4:
                    continue
            if count>50:
                break
            sessions.append(record['session'])
            count += 1
            print(count)

            #break  # 我去掉了这里的 break，因为你可能想处理整个数据
            strresult = ''
            listresult = []
            prompt_tmp = """\你是一个文本提取专家，请你根据我的要求，正确的提取内容。需求1.请你精确的提取这段对话中的{outputkey}信息。需求2.如果是卡号，身份证号，手机号码则都必须是纯数字，名字是纯中文。需求3.按照指定格式提取信息，keyvalue格式.不需要回复多余的信息，只需要提取出来的信息。请解析对应的信息:"""
            prompt = PromptTemplate.from_template(prompt_tmp)
            system_content = prompt.format(outputkey=outputkeylist)
            user_content = str(record['session'])
            assistant_content = str(record['output'])
            result_data = construct_fake_data(system_content, user_content, assistant_content)

            # 将 result_data 转化为 JSON 格式字符串，并写入到文件中
            outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')


import itertools
import time
import os
import re
current_time = time.time()


def chatgpt_tanslate(prompt, chatgpt_api):
    global current_time
    gap_time = time.time() - current_time    #防止api调用频率过高，可根据RPM限制进行调整
    if gap_time < 20:
        time.sleep(20-gap_time)
    current_time = time.time()

    from openai import OpenAI
    client = OpenAI(api_key=chatgpt_api)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        #{"role": "system", "content": None},
        {"role": "user", "content": prompt},
        #{"role": "assistant", "content": None},
    ]
    )
    return completion.choices[0].message.content


def is_time_format(s):
    pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
    return bool(re.match(pattern, s))


def srt_format(srt_path, n, is_initial_sub = 0): #n: 每n-1行添加一个空行
    formatted_lines = []
    with open(srt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].split('\n')[0]
            if is_time_format(line):       #判断字幕时间格式
                flag_jump = False
                if is_initial_sub and  i+2 < len(lines):
                    flag_jump = is_time_format(lines[i+2].split('\n')[0]) or is_time_format(lines[i+2].split('\n')[0]) or lines[i+1] == '\n' or lines[i+2] == '\n'
                if flag_jump:       #翻译为空且与下下一个字幕直接相连时跳过格式处理，以默认后面重新翻译
                    print('\njump', lines[i-1].split('\n')[0])
                    continue
                for j in range(-1,n-2):
                    if lines[i+j] == '\n':    #判断字幕是否为空
                        for k in range(n-2-j):
                            formatted_lines.append('\n')
                        break
                    formatted_lines.append(lines[i+j])
    with open(srt_path, 'w', encoding='utf-8') as file:
        i = 1
        for line in formatted_lines:
            file.write(line)
            if i % (n-1) == 0:
                file.write('\n')
            i += 1


def read_lines_range(file_path, start_num, end_num):
    with open(file_path, 'r', encoding='utf-8') as file:
        missing_txt = ""
        for i in range(1,end_num):    
            next_four_lines = list(itertools.islice(file, 4))
            if i > start_num:
                for line in next_four_lines:
                    missing_txt += line
    return missing_txt


def retranslate(srt_path, srt_translated_path, chatgpt_api, prompt_ask, srt_number):
    with open(srt_translated_path, 'r+', encoding='utf-8') as file:
        srt_tanslate = file.readlines()
        falg_change = False
        i = 0
        j = 0
        k = int(srt_tanslate[i+5].split('\n')[0])
        while i+5 < len(srt_tanslate) or i+5 < srt_number*5:
            if i+5 == len(srt_tanslate):    #字幕末尾遗漏翻译
                j = int(srt_tanslate[i].split('\n')[0])
                k = srt_number + 1
            else:       #字幕中间遗漏翻译
                j = int(srt_tanslate[i].split('\n')[0])
                k = int(srt_tanslate[i+5].split('\n')[0])
            if j+1 != k:
                missing_txt = read_lines_range(srt_path, j, k)
                prompt = prompt_ask + missing_txt
                print(prompt + "\n" + "<< retanslating")
                respond = chatgpt_tanslate(prompt, chatgpt_api)
                with open('temp.srt', 'w', encoding='utf-8') as file:
                    file.write(respond+'\n\n')
                    srt_format('temp.srt',5)
                with open('temp.srt', 'r', encoding='utf-8') as file:
                    respond_foramt = file.readlines()
                os.remove('temp.srt')
                l = i+5
                for line in respond_foramt:
                    srt_tanslate.insert(l, line)
                    l += 1
                falg_change = True
            i += 5
    if falg_change:
        with open(srt_translated_path, 'w', encoding='utf-8') as file:
            file.writelines(srt_tanslate)


def main(srt_path, srt_translated_path, chatgpt_api, prompt_path, srt_number):
    srt_format(srt_path, 4)
    if os.path.exists(srt_translated_path):       #if srt file exists, skip
        return
    with open(srt_translated_path, 'w', encoding='utf-8') as file:
        pass
    with open(srt_path, 'r', encoding='utf-8') as file1, open(srt_translated_path, 'r+', encoding='utf-8') as file2, open(prompt_path, 'r', encoding='utf-8') as file3:
        flag = True
        srt_split = ""
        prompt_ask = file3.read()
        while flag:
            for i in range(50):    #每次读取的最大字幕行数
                next_four_lines = list(itertools.islice(file1, 4))
                if not next_four_lines:
                    flag = False
                    break
                for line in next_four_lines:
                    srt_split += line
                if len(srt_split) > 2500:   #每次输入的最大字幕字数长度
                    break
            prompt = prompt_ask + srt_split
            print(prompt + "\n" + "<< translating")
            respond = chatgpt_tanslate(prompt, chatgpt_api)
            file2.write(respond + "\n\n")
            srt_split = ""
    srt_format(srt_translated_path, 5, 1)
    retranslate(srt_path, srt_translated_path, chatgpt_api, prompt_ask, srt_number)
    srt_format(srt_translated_path, 5)

    #检验翻译是否完整
    with open(srt_translated_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0,len(lines),5):
            j = int(lines[i].split('\n')[0])
            if j*5 == len(lines)-5:
                print("翻译完整")
                break
            elif j+1 != int(lines[i+5].split('\n')[0]):
                print("翻译不完整")
                break
    time.sleep(3)
## input : config.whisperFile
## output : config.AIPunctuationFile

from openai import OpenAI
import os
import re
import json
import time
import config_reader

config = config_reader.config

inputfile = config.whisperFile
process_path = os.path.join("process")

with open(os.path.join(process_path,config.AIPunctuationLog), 'w', encoding='utf-8') as file:
    file.write('')

debug_text = ""

def time_to_seconds(time_str):
    """将时间字符串转换为秒"""
    hours, minutes, seconds_milliseconds = time_str.split(':')
    if ',' in seconds_milliseconds:
        seconds, milliseconds = map(int, seconds_milliseconds.split(','))
    else:
        seconds = int(seconds_milliseconds)
        milliseconds = 0

    return int(hours) * 3600 + int(minutes) * 60 + seconds + milliseconds / 1000000

def parse_srt(file_content):
    """解析 SRT 文件内容"""
    pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2}(?:,\d{6})?) --> (\d{1,2}:\d{2}:\d{2}(?:,\d{6})?)\n(.*?)\n\n', re.DOTALL)
    subtitles = []

    for match in pattern.finditer(file_content):
        #print(match)
        start_str, end_str, text = match.groups()
        # 確保不是空字串再轉存
        if len(text.replace('\n', ' ')) > 0:
            start = time_to_seconds(start_str)
            end = time_to_seconds(end_str)
            subtitles.append({
                "id": len(subtitles) + 1,
                "start": start,
                "end": end,
                "text": text.replace('\n', ' ')
            })

    return subtitles

def read_srt_file(filename):
    """从文件中读取 SRT 内容"""
    with open(os.path.join(process_path,filename), 'r', encoding='utf-8') as file:
        return file.read()

# 读取字幕文件
srt_content = read_srt_file(inputfile)

# 解析字幕文件
subtitles = parse_srt(srt_content)

def check_subtitles(subtitles):
    # 新数组，用于存储连续的不符合条件的字幕对象
    continuous_subtitles = []
    if not config.lineArrange:
        return continuous_subtitles
    # 用于计数连续不符合条件的字幕对象数量
    count = 0
    count_flag = False


    for subtitle in subtitles:
        text = subtitle['text']

        #是否已進入連續狀態
        if count_flag:
            #字句仍然連續 繼續加入處理陣列
            if not (text.endswith('.') or text.endswith('!') or '. ' in text):
                continuous_subtitles.append(subtitle)
            else:
                #不連續則中斷並改回flag
                count_flag = False
        else:
            if not (text.endswith('.') or text.endswith('!') or '. ' in text):
                #發現開始有無標點字串 先加入處理陣列
                continuous_subtitles.append(subtitle)
                #已經超過三句連續 進入正式連續狀態
                if count > 3:
                    count = 0
                    count_flag = True
                else:
                    #已經未過三句連續 繼續加總閾值
                    count += 1
            else:
                #發現斷點 而且已經處於連續狀態
                if count_flag:
                    count_flag = False
                else:
                    #發現斷點 但最後加進的字串未達連續閾值 全回溯清除
                    while count > 0:
                        continuous_subtitles.pop()
                        count -= 1

    return continuous_subtitles


def group_array(arr):

    # 分组
    groups = []
    current_group = []

    for item in arr:
        if not current_group:
            # 如果当前组为空，直接添加元素
            current_group.append(item)
        else:
            # 检查当前元素的 id 是否与前一个元素的 id 连续
            if item['id'] - current_group[-1]['id'] == 1:
                current_group.append(item)
            else:
                # 如果不连续，将当前组添加到 groups，然后开始新的组
                groups.append(current_group)
                current_group = [item]

    # 添加最后一个组（如果有）
    if current_group:
        groups.append(current_group)

    # 過長的連續字串需分批切割餵給API
    split_index = 13
    index = 0
    while True:
        if len(groups[index]) >= 16:
            cur_ele = groups.pop(index)
            groups.insert(index,cur_ele[:split_index])
            groups.insert(index+1,cur_ele[split_index:])
        index += 1
        #掃描完畢    
        if index >= len(groups):
            break

    return groups

##########################################處理文句過長沒分段問題############################################### 
with open(os.path.join(process_path,config.workingFile), 'r') as file:
    lng = json.load(file)['lng']

# 示例：应用函数并显示结果
if config.lineArrange and lng == 'en':
    print(f"檢查標點符號是否缺失(僅限英文字句)")
    problematic_subtitles = check_subtitles(subtitles)
else:
    problematic_subtitles = []

if problematic_subtitles:
    grouped_objects = group_array(problematic_subtitles)

    for group in grouped_objects:
        print([obj['id'] for obj in group])

    if len(grouped_objects) == 1 and not grouped_objects[0]:
        grouped_objects = []
else:
    grouped_objects = []

###########################################讓AI處理文句分段問題####################################################

with open('moduel/API.key', 'r') as file:
    key = file.read()
client = OpenAI(
    api_key=key,
)

# 步骤 1: 创建一个助手
assistant = client.beta.assistants.create(
    name="John Doe",
    instructions="你是個翻譯，負責翻譯外國字幕轉成文句通暢的中文",
    model=config.splitLineModel
)

def update_subtitles(subtitles, adj_sentences):

    for sentence in adj_sentences:
        # 分割每行以获取开始时间和文本
        if "##" in sentence:
            token = sentence.split("##")
            sst = token[0].strip()

            # 在subtitles中查找匹配的开始时间
            for subtitle in subtitles:
                if str(subtitle["start"]) == str(sst):
                    # 找到匹配项，更新text并保存旧的text
                    subtitle["text_old"] = subtitle["text"]
                    subtitle["text"] = token[-1]
                    break

def evalResult(text, lineNum):
    print("*****資料合法性解析*****")
    returnLines = []
    for line in text:
        cleaned_line = line.strip('"')
        pattern = r"^(\d+(\.\d{1,2})?)[^a-zA-Z]*"
        match = re.match(pattern, line)
        if match:
            fixed_line = re.sub(pattern, r"\1##", cleaned_line)
            print(cleaned_line + "  匹配完成")
            returnLines.append(fixed_line)
        else:
            print(cleaned_line + "  無匹配")
    return returnLines

for idx, group in enumerate(grouped_objects, start=1):

    passFlag = True
    currentReq = 0
    tempReault = []
    while passFlag:
        currentReq += 1
        # 构建消息内容，包括字幕文本
        subtitles_text = "\n".join([f"{subtitle['start']}##{subtitle['text']}" for subtitle in group])
        message_content = "本句之後是一段字幕內容 ##前面的數值與##本身不須更動並且必須保留 觀察每句的前後文然後將文章補上標點符號 另外注意每行的結尾不一定是整句的結尾,請仔細觀察前後文的內容再決定句號的位置. 另外不要使用省略號 \n" + subtitles_text
        print(f"############################################# {idx} / {len(grouped_objects)} ###############################################")
        # print(f"{message_content}")
        print(f"{subtitles_text}")
        print(f"--------------------------------------------- {idx} / {len(grouped_objects)} -----------------------------------------------")

        debug_text += f"############################################# {idx} / {len(grouped_objects)} ###############################################\n"
        debug_text += subtitles_text + "\n"
        debug_text += f"--------------------------------------------- {idx} / {len(grouped_objects)} -----------------------------------------------\n"

        # 步骤 2: 创建一个线程
        thread = client.beta.threads.create()

        # 步骤 3: 向线程添加一条消息
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message_content
        )

        # 步骤 4: 运行助手
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            # instructions="Please address the user as Jane Doe. The user has a premium account."
            instructions="這些句子沒有適當的標點符號 請觀察每句的前後文然後在適合的文句補上標點符號(注意! 每行的結尾不一定是整句的結尾,請仔細觀察前後文的內容) 讓每句子盡量不超過20個單字.另外不要使用省略號"
        )

        # 检查运行状态
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == 'completed':
                break
            time.sleep(1)  # 等待一秒再次检查

        # 步骤 5: 显示助手的回应并处理结果
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for message in messages.data:
            if message.role == "assistant":
                for content in message.content:
                    if content.type == 'text':
                        # 顯示斷句內容
                        text = content.text.value
                        text = text.replace('\\(', '(').replace('\\)', ')')
                        # print(text)
                        
                        adj_text = content.text.value
                        adj_sentences = [line for line in adj_text.split('\n') if line.strip()]

                        debug_text += text + "\n" + f"建議行數 : {len(group)}\n"
                        print(adj_text + f"\n建議行數 : {len(group)}\n")
                        fixed_sentences = evalResult(adj_sentences,len(group))
                        print("...................................................................")
                        for sentence in fixed_sentences:
                            print(sentence)
                        print(f"\n合法行數 : {len(fixed_sentences)}\n")
                        #update_subtitles(subtitles, adj_sentences)
                        if len(fixed_sentences) == len(group):
                            update_subtitles(subtitles, fixed_sentences)
                            passFlag = False
                            break
                        else:
                            if currentReq > config.maxReqTime:
                                update_subtitles(subtitles, max(tempReault, key=len))
                                passFlag = False
                                break
                            else:
                                tempReault.append(fixed_sentences)
                            continue


def split_subtitle(subtitle):
    """根据句号分割字幕，并调整时间戳"""
    text = subtitle['text']
    start = subtitle['start']
    end = subtitle['end']

    # 检查文本是否包含句号
    if '. ' in text or '! ' in text or '? ' in text or '。' in text:
        # 根据句号分割文本
        sentences = re.split(r'(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bMs)\. |[!?。] ', text)
        split_subtitles = []

        # 计算每个句子的时间戳
        total_length = len(text)
        current_start = start

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            # 确保不除以零
            if total_length > 0 and sentence_length > 0:
                proportion = sentence_length / total_length
                current_end = current_start + proportion * (end - start)

                # 为除最后一个句子外的句子添加句号
                if i < len(sentences) - 1:
                    sentence += '.'

                # 创建新的字幕对象
                split_subtitles.append({
                    "start": current_start,
                    "end": current_end,
                    "text": sentence.strip()
                })

                current_start = current_end

        return split_subtitles
    else:
        return [subtitle]

def is_end_of_sentence(text):
    # 檢查文本是否以非縮寫的句號、驚嘆號或問號結尾
    return re.search(r'(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bMs)(?<!\bSt)\.(?!\w)|[!?。]$', text) is not None


def merge_subtitles(subtitles, max_length=config.maxCharPerLine):
    merged_subtitles = []
    i = 0

    if not config.lineArrange:
        return subtitles

    while i < len(subtitles):
        current_subtitle = subtitles[i]
        text = current_subtitle['text']
        start = current_subtitle['start']
        end = current_subtitle['end']

        # 合併直到找到以句號結尾的文本或超過最大長度
        while not is_end_of_sentence(text) and i < len(subtitles) - 1:
            next_subtitle = subtitles[i + 1]
            next_text = next_subtitle['text']

            # 檢查合併後的長度
            if len(text + ' ' + next_text) > max_length:
                break

            # 更新字幕信息
            text += ' ' + next_text
            end = next_subtitle['end']
            i += 1

        # 添加合併後的字幕
        merged_subtitles.append({
            "start": start,
            "end": end,
            "text": text
        })
        i += 1

    return merged_subtitles


def round_timestamps(subtitles):
    rounded_subtitles = []

    for subtitle in subtitles:
        rounded_start = round(subtitle['start'], 2)
        rounded_end = round(subtitle['end'], 2)
        rounded_subtitles.append({
            "start": rounded_start,
            "end": rounded_end,
            "text": subtitle['text']
        })

    return rounded_subtitles

# 应用分割函数
if config.lineArrange and lng == 'en':
    split_subtitles = []
    for subtitle in subtitles:
        split_subtitles.extend(split_subtitle(subtitle))

    # 应用合并函数
    merged_subtitles = merge_subtitles(split_subtitles)


    # 应用四舍五入函数
    rounded_subtitles = round_timestamps(merged_subtitles)
else:
    rounded_subtitles = subtitles

with open(os.path.join(process_path,config.AIPunctuationLog), 'w', encoding='utf-8') as file:
    file.write(debug_text)

# 将处理后的字幕数据写入 JSON 文件
with open(os.path.join(process_path,config.AIPunctuationFile), 'w', encoding='utf-8') as json_file:
    json.dump(rounded_subtitles, json_file, ensure_ascii=False, indent=4)

print(f"字幕数据已保存到 {config.AIPunctuationFile} 文件中。")

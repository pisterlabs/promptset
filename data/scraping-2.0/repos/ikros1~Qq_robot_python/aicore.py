from role_set import role_dict
import json
import os
import time
import concurrent.futures

from google_search import google_search_to_json, extract_info_from_json, split_text

from web_analyze import analyze_text, clean_text
# from text_to_wav_interface import display_images
import openai
from dotenv import load_dotenv
from tool_kit import send_message_to_group, is_similar

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

moji_dict = {'平和': "/ts", '开心': '/gz', '惊讶': '/fd', '欣慰': '/hanx',
             '愉悦': '/ww', '尴尬': '/lengh', '生气': '/lyj', '悲伤': '/dk',
             '惆怅': '/cs', '害羞': '/hx', '疑惑': '/yiw'}


# 角色扮演预设


def internet_analyze_front_prompt():
    # 直接创建列表可能出错字数太多，所以先创建一个空字典，再将字典添加到列表中，好坑
    prompt_list = []
    prompt_list.append({"role": "system",
                        "content": "你是人工智能助手，人工智能助手可以连接网络查询内容，回答的内容以json的格式返回，回答字符除了json整体外禁止添加任何内容，json的格式如下{\"keywords\":\"这个键值对的值里面填入人工智能助手想要在互联网上查找回答需要的信息的关键字，如果没有请置空\",\"title\":\"当我给出查询信息时这里面填入你想要了解全文内容的title\",\"search_flag\":\"如果人工智能需要在互联网查询内容请将这个键值对的值设为1，不需要则为0\",\"view_details_flag\":\"如果上述对话中提供的搜索记录中有可以解决问题信息的标题，人工智能需要浏览标题所指向的文章的详细内容请将这个键值对的值设为1，不需要则为0\",\"answer\":\"这个键值对的值里面填入人工智能助手的回答\"}"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"当前暂时没有需要回答的问题，关键词为空\",\"title\":\"暂无需要查看文本的标题\",\"search_flag\":\"0\",\"view_details_flag\":\"0\",\"answer\":\"人工智能助手理解了系统要求并在接下来的对话中依规定的格式回答，请提出你的问题\"}"})
    prompt_list.append({"role": "user", "content": "用户12317437说：猫猫乱叫怎么办"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"猫乱叫\",\"title\":\"暂无需要查看文本的标题\",\"search_flag\":\"1\",\"view_details_flag\":\"0\",\"answer\":\"正在联网查询\"}"})
    prompt_list.append({"role": "user",
                        "content": "(True, [{'title': '猫 - Wiktionary', 'url': 'https://en.wiktionary.org/wiki/%E7%8C%AB', 'snippet': 'For pronunciation and definitions of 猫 – see 貓 (“cat; to hide oneself; etc.”). (This character, 猫, is the simplified and variant form of 貓.) Notes:.'}])"})
    prompt_list.append({"role": "user", "content": "人工智能助手上述是按你要求在互联网上获得最新的实时信息，包含标题，文章链接，文本切片。你需要从中提取出你想要的信息来回答问题。"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"了解全文细节中不需要关键词\",\"title\":\"猫 - Wiktionary\",\"search_flag\":\"0\",\"view_details_flag\":\"1\",\"answer\":\"正在了解全文\"}"})
    prompt_list.append({"role": "user",
                        "content": "详细信息为：猫是一种家养小型食肉哺乳动物，属中型猫科动物。根据遗传学及考古学分析，人类养猫的纪录可追溯至10,000年前的新月沃土地区，古埃及人饲养猫的纪录可追溯至公元前1000年前，以防止老鼠吃掉谷物。现在，猫成为世界上最为广泛的宠物之一，饲养率仅次于狗，但同时也威胁着很多原生鸟类。"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"猫乱叫\",\"title\":\"暂无需要查看文本的标题\",\"search_flag\":\"1\",\"view_details_flag\":\"0\",\"answer\":\"上文没有足够信息回答问题，再次联网查询\"}"})
    prompt_list.append({"role": "user",
                        "content": "(True, [{'title': '猫 - Wiktionary', 'url': 'https://en.wiktionary.org/wiki/%E7%8C%AB', 'snippet': 'For pronunciation and definitions of 猫 – see 貓 (“cat; to hide oneself; etc.”). (This character, 猫, is the simplified and variant form of 貓.) Notes:.'},{'title': '猫乱叫- 维基百科，自由的百科全书', 'url': 'https://zh.wikipedia.org/wiki/%E7%8C%AB', 'snippet': '貓是一種胎生動物。在发情期（主要是冬末至夏初，但許多貓可以長年發情），公貓到處撒尿，而母貓在半夜狂吼亂叫，民間俗稱此為“鬧貓”或“猫叫春”。通常經由結紮手術可解決或\xa0...'}])"})
    prompt_list.append({"role": "user", "content": "人工智能助手上述是按你要求在互联网上获得最新的实时信息，包含标题，文章链接，文本切片。你需要从中提取出你想要的信息来回答问题。"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"了解全文细节中不需要关键词\",\"title\":\"猫乱叫- 维基百科，自由的百科全书\",\"search_flag\":\"0\",\"view_details_flag\":\"1\",\"answer\":\"正在了解全文\"}"})
    prompt_list.append({"role": "user",
                        "content": "详细信息为：住户家附近若有饲养猫的邻居，因未结扎发情(常当作理由)或其他原因，且饲主无法做好隔音处理，叫声尖锐扰民，影响邻居住家品质。如果猫感到饥饿或口渴，它们可能会大声叫喊以吸引注意力并表达自己的需求。"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"不需要关键词\",\"title\":\"无\",\"search_flag\":\"0\",\"view_details_flag\":\"0\",\"answer\":\"结合上文，猫乱叫的原因可能有多种情况，以下是一些常见的原因：饥饿或口渴：如果猫感到饥饿或口渴，它们可能会大声叫喊以吸引注意力并表达自己的需求。\"}"})

    return prompt_list


def robot_talk_front_prompt(key):
    # 直接创建列表可能出错，所以先创建一个空字典，再将字典添加到列表中，好坑
    prompt_list = []
    role_string = role_dict[key]
    prompt_list.append({"role": "system",
                        "content": role_string})

    return prompt_list


def wisper(file):
    audio_file = open(file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    json_obj = json.loads(str(transcript))

    # 获取文本值
    text = json_obj['text']
    return text


def moji_find(txt):
    sentence = txt
    emotions = sentence.split("情感：")[-1]
    for emotion in moji_dict:
        if emotions.find(emotion) != -1:
            return moji_dict[emotion]
    return moji_dict['平和']


def easy_core(input_text):
    prompt_list = []
    prompt_list.append({"role": "system",
                        "content": "你是人工智能助手，你需要回答试卷上的问题，下面为你需要回答的问题的试卷经过ocr扫描后的文本，你需要从中提取出其中的问题并回答问题,题目可能是个陈述句，可能丢失部分符号，你需要分析其中逻辑返回答案。"})
    prompt_list.append({"role": "user",
                        "content": "ocr扫描后详细信息为：" + input_text})
    prompt_list.append({"role": "user",
                        "content": "请回答试卷上的问题，先给出答案再解释答案的原因。"})

    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=prompt_list,
            temperature=0
        )
    # 收集异常，打印异常
    except Exception as e:
        print(e)
        return False, "openai连接失败"

    return True, str(response.choices[0].message.content)


def cool_core(input_list):
    prompt_list = []
    for i in input_list["assistant_id"]:
        prompt_list.append(i)
    for i in input_list["assistant_memory"]:
        prompt_list.append(i)
    print("调用联网核心" + str(prompt_list) + "\n")
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=prompt_list,
            temperature=0
        )
    # 收集异常，打印异常
    except Exception as e:
        print(e)
        return False, "openai连接失败"

    return True, str(response.choices[0].message.content)


def warm_core(input_list):
    prompt_list = []
    for i in input_list["ikaros_id"]:
        prompt_list.append(i)
    for i in input_list["group_memory"]:
        prompt_list.append(i)
    print("调用本地核心" + str(prompt_list) + "\n")
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=prompt_list,
            temperature=1
        )
    except Exception as e:
        print(e)
        return False, "openai连接失败"

    return True, str(response.choices[0].message.content)


def process_text(txt_info, i, key_words, store_list, flag_list):
    index = i
    prompt_list_info = {"assistant_id": [], "assistant_memory": [{"role": "user",
                                                                  "content": "人工智能助手，接下来会提供一段爬取来自网页的文本，主题是关于:" + key_words + "，你需要简化信息，提取和主题相关的主要信息，删除与主题无关的信息，简化后的文本会交给gpt3.5用于文本分析用，请采取gpt3.5容易读取并字数较少的回答方式，在回答内容中不要有除了简化后文本以外的其他内容。文本如下：" + str(
                                                                      txt_info)}]}
    success, response_str = cool_core(prompt_list_info)
    store_list[index] = response_str
    flag_list[index] = True
    return success


def simplified_txt(txt, length, key_words):
    txt = clean_text(txt)
    if len(txt) > length:
        print("长度大于标准开始简化,长度为： " + str(len(txt)))
        return_txt = txt
        while len(return_txt) > length:
            temp_str = return_txt
            divided_txt_list = split_text(temp_str)
            return_txt = ""
            store_list = ["" for i in range(len(divided_txt_list))]
            flag_list = [False for i in range(len(divided_txt_list))]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(len(divided_txt_list)):
                    print("开启线程： " + str(i))
                    futures.append(
                        executor.submit(process_text, divided_txt_list[i], i, key_words, store_list, flag_list))
                concurrent.futures.wait(futures)
            for i in range(len(flag_list)):
                while not flag_list[i]:
                    time.sleep(0.1)

            for i in store_list:
                return_txt = return_txt + i

            return_txt = clean_text(return_txt)

    else:
        return_txt = txt

    return_txt = clean_text(return_txt)
    print("简化完成 内容为" + return_txt + "长度为" + str(len(return_txt)))
    return return_txt


def google_search_key(key_worlds):
    success, json = google_search_to_json(key_worlds)
    trans_success = False
    if success:
        trans_success, info = extract_info_from_json(json)
    else:
        info = []
    return success, trans_success, info


def answer_correct(response_json_str, input_list):
    temp_list = []
    # 复制input_list内容，防止修改原内容

    response_json = {"search_flag": "0", "keywords": "无", "view_details_flag": "0", "title": "无", "answer": "无法解析的回答"}
    re_str = response_json_str
    for i in range(3):
        try:
            response_json = json.loads(re_str)
            if "search_flag" in response_json and "keywords" in response_json and "answer" in response_json and "view_details_flag" in response_json and "title" in response_json:
                break
        except json.JSONDecodeError:

            input_list["assistant_memory"].append({"role": "assistant", "content": str(response_json_str)})
            print("人工智能助手未按规定的格式回答")
            input_list["assistant_memory"].append({"role": "user", "content": "人工智能助手未按之前对话中的规范格式回答，请改正你的回答"})
            # 在这里处理异常情况，比如输出错误信息或者返回一个默认值
            success, re_str = cool_core(input_list)
            if not success:
                response_json = {"search_flag": "0", "keywords": "无", "view_details_flag": "0", "title": "无",
                                 "answer": "无法解析的回答"}
                break
            else:
                print("openai_api_return:  " + response_json_str)
            if i == 2:
                response_json = {"search_flag": "0", "keywords": "无", "view_details_flag": "0", "title": "无",
                                 "answer": "无法解析的回答"}
                break
    return response_json


class AnswerLoop:
    def __init__(self):
        self.input_list = {}
        self.from_group = ""
        self.key_worlds = ""
        self.title = ""
        self.search_api_memory = []

    def data_set(self, input_list, from_group):
        self.input_list = input_list
        self.from_group = from_group
        self.key_worlds = ""
        self.search_api_memory = []
        self.title = ""

    def run(self):
        success, response_json_str = cool_core(self.input_list)
        if not success:
            print("openai_api_return:  " + response_json_str)
            send_info = {'type': 'text',
                         'data': {'text': "openai连接失败，请联系管理员"}}
            send_message_to_group(self.from_group, send_info)
            return
        response_json = answer_correct(response_json_str, self.input_list)
        if response_json["search_flag"] == "1" and response_json["view_details_flag"] == "0":
            self.input_list["assistant_memory"].append({"role": "assistant", "content": str(response_json_str)})
            self.key_worlds = response_json["keywords"]
            success = self.deal_search_api_memory_list()
            if success:
                self.run()
            else:
                send_info = {'type': 'text',
                             'data': {'text': "谷歌搜索连接失败，请联系管理员"}}
                send_message_to_group(self.from_group, send_info)
        elif response_json["view_details_flag"] == "1" and self.key_worlds != "":
            self.input_list["assistant_memory"].append({"role": "assistant", "content": str(response_json_str)})
            self.title = response_json["title"]
            self.deal_web_txt()
            self.run()
        else:
            send_info = {'type': 'text',
                         'data': {'text': str(response_json["answer"])}}
            send_message_to_group(self.from_group, send_info)

    def deal_search_api_memory_list(self):
        search_success, turn_success, info = google_search_key(self.key_worlds)
        if search_success and turn_success:
            for data in info:
                self.search_api_memory.append(data)
            self.input_list["assistant_memory"].append({"role": "user", "content": "信息为" + str(info)})
            self.input_list["assistant_memory"].append(
                {"role": "user", "content": "人工智能助手上述是按你要求在互联网上获得最新的实时信息，包含标题，文章链接，文本切片。你需要从中提取出你想要的信息来回答问题。"})
            return True

        elif not search_success:
            print("谷歌连接失败")
            self.input_list["assistant_memory"].append({"role": "user", "content": "谷歌连接失败，请检查网络连接"})
            return False
        else:
            print("谷歌搜索无结果")
            self.input_list["assistant_memory"].append({"role": "user", "content": "谷歌搜索无结果，请更换关键词重新提问"})
            return True

    def deal_web_txt(self):
        max = 0
        url_t = ""
        for dict_data in self.search_api_memory:
            if is_similar(self.title, dict_data['title']) >= max:
                max = is_similar(self.title, dict_data['title'])
                url_t = dict_data['url']
        if max > 0.75:
            web_info = analyze_text(url_t)
            web_info = simplified_txt(web_info, 500, self.key_worlds)
            self.input_list["assistant_memory"].append(
                {"role": "user", "content": "具体信息经过提取如下：" + web_info + "其中部分内容可能与问题无关，请忽略,结合上述信息请回答。"})
            return True
        else:
            self.input_list["assistant_memory"].append({"role": "user", "content": "具体信息没有找到,请重新提问。"})
            return False

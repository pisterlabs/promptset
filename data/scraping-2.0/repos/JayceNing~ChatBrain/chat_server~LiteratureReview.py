import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
from collections import OrderedDict
import textwrap
import numpy as np
import openai

################################
# 词云图
################################
def split_by_language(input_list):
    chinese_list = []
    english_list = []
    
    for item in input_list:
        if any(char >= '\u4e00' and char <= '\u9fff' for char in item):
            chinese_list.append(item)
        else:
            english_list.append(item)
    
    return chinese_list, english_list

def extract_keywords_from_md(md_content):
    keywords_flags = ['关键词', '关键字']
    keywords_list = []
    
    for keywords_flag in keywords_flags:
        keywords_match = re.search(f'{keywords_flag}[：:](.*?)\n', md_content)
        if keywords_match:
            keywords_str = keywords_match.group(1)
            keywords_str = re.sub(r'[\(\)（）【】]', ',', keywords_str)
            # 使用正则表达式匹配多种分隔符并切分关键词
            keywords = re.split(r'[,;；、，·]', keywords_str)
            cleaned_keywords = []
            for keyword in keywords:
                cleaned_keywords.append(keyword.strip())
            keywords_list.extend(cleaned_keywords)
            
    keywords_list = [item for item in keywords_list if item != '']
    return split_by_language(keywords_list)

def merge_and_count_elements(input_list):
    element_counts = {}
    element_cased = {}  # 用于保存大小写形式
    
    for element in input_list:
        element_lower = element.lower()  # 将元素转换为小写
        element_stripped = element.rstrip("s")  # 去除末尾的"s"
        
        if element_lower in element_counts:
            element_counts[element_lower] += 1
        elif element_stripped.lower() in element_counts:  # 检查去除"s"后的形式
            element_counts[element_stripped.lower()] += 1
        else:
            element_counts[element_stripped.lower()] = 1

        if element_lower != element:  # 大写形式的元素
            element_cased[element_stripped.lower()] = element

    
    merged_result = []
    for element, count in element_counts.items():
        if element in element_cased:
            merged_result.append((element_cased[element], count))
        else:
            merged_result.append((element, count))
    
    # 按照频次从高到低排序
    merged_result.sort(key=lambda x: x[1], reverse=True)
    
    return merged_result



def plot_wordcloud(data, font_path, save_path):
    if font_path == 'none':
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(data))
    else:
        wordcloud = WordCloud(width=1600, height=800, background_color='white', font_path=font_path).generate_from_frequencies(dict(data))

    if os.path.exists(save_path):
        # 删除文件
        os.remove(save_path)
    wordcloud.to_file(save_path)


def get_all_keyword_list_and_generate_wordclouds(paths, useremail):
    all_keywords_list_zh = []
    all_keywords_list_en = []
    for path in paths:
        # 读取 Markdown 文件内容
        md_file_path = path + '/md.md'  # 替换为你的 Markdown 文件路径
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 提取关键字并保存到列表
        chinese_list, english_list = extract_keywords_from_md(md_content)
        for keyword in chinese_list:
            all_keywords_list_zh.append(keyword)
        for keyword in english_list:
            all_keywords_list_en.append(keyword)

    res_zh = merge_and_count_elements(all_keywords_list_zh)
    current_path = os.getcwd()
    font_path = current_path + '/SimHei.ttf'

    save_folder = os.getcwd() + '/literatureReview/' + useremail + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    success_zh = 0
    success_en = 0
    try:
        plot_wordcloud(res_zh, font_path, save_folder + 'wordcloud_zh.png')
        success_zh = 1
    except:
        pass
    
    res_en = merge_and_count_elements(all_keywords_list_en)
    try:
        plot_wordcloud(res_en, font_path, save_folder + 'wordcloud_en.png')
        success_en = 1
    except:
        pass
    
    return success_zh, success_en, res_zh, res_en

################################
# 关键词年份
################################

# 绘制每年出版物数量
def cul_paper_num_per_month(data):
    # 创建一个空的字典来存储每年的关键词数量
    yearly_frequency = {}

    # 遍历字典中的每个项
    for entry in data:
        time = entry[0]  # 获取时间
        keywords = entry[1]  # 获取关键词列表

        # 提取年份
        year = time.split('-')[0]

        # 将关键词数量与年份关联起来
        if year in yearly_frequency:
            yearly_frequency[year] += 1
        else:
            yearly_frequency[year] = 1

    # 删除时间为空的项
    if '' in yearly_frequency:
        del yearly_frequency['']
    
    sorted_dict = dict(sorted(yearly_frequency.items()))
    
    return sorted_dict

def draw_pub_num_per_year_bar(pubulication_num_per_year, Publication_Area, useremail):
    current_path = os.getcwd()
    font_path = current_path + '/SimHei.ttf'
    font = FontProperties(fname=font_path, size=14)

    # 示例数据
    categories = list(pubulication_num_per_year.keys())
    values = list(pubulication_num_per_year.values())
    #Publication_Area = 'Brain computer interface'

    fig, ax = plt.subplots(figsize=(8, 4))
    # 去除图表边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 添加横辅助线
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # 绘制柱状图
    plt.bar(categories, values)
    # 添加标题和标签
    title = f'Number of Publications in {Publication_Area}'
    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    plt.suptitle(wrapped_title)
    # plt.title(f'Number of Publications in {Publication_Area}')
    plt.xticks(rotation=90, fontproperties=font)
    plt.ylabel('')
    plt.xlabel('Year', fontproperties=font)

    save_folder = os.getcwd() + '/literatureReview/' + useremail + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = save_folder + 'NumOfPublicationPerYear.png'

    # 显示图形
    plt.tight_layout()  # 调整图形布局
    plt.savefig(save_path)

    return save_path

def extract_info_from_md(md_content):
    time_pattern = r'(\d{4}-\d{2})'
    keywords_flags = ['关键词', '关键字']
    keywords_list = []
    time = ''
    
    time_match = re.search(time_pattern, md_content)
    if time_match:
        time = time_match.group(1)
    
    for keywords_flag in keywords_flags:
        keywords_match = re.search(f'{keywords_flag}：(.*?)\n', md_content)
        if keywords_match:
            keywords_str = keywords_match.group(1)
            keywords_str = re.sub(r'[\(\)（）【】]', ',', keywords_str)
            keywords = re.split(r'[,;；、，·]', keywords_str)
            cleaned_keywords = []
            for keyword in keywords:
                cleaned_keywords.append(keyword.strip())
            keywords_list.extend(cleaned_keywords)
    
    keywords_list = [item for item in keywords_list if item != '']
    return time, keywords_list

def process_md_files_in_folder(folder_path):
    md_files_info = []
    
    for file_name in os.listdir(folder_path):
        file_name = file_name + '/md.md'
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()
            time, keywords = extract_info_from_md(md_content)
            md_files_info.append((time, keywords))
    
    return md_files_info

def merge_keywords_by_time(md_files_info):
    merged_keywords = OrderedDict()  # 使用有序字典来保持按照时间排序
    
    for time, keywords in md_files_info:
        if time not in merged_keywords:
            merged_keywords[time] = []
        merged_keywords[time].extend(keywords)
    
    return merged_keywords

def split_by_language(input_list):
    chinese_list = []
    english_list = []
    
    for item in input_list:
        if any(char >= '\u4e00' and char <= '\u9fff' for char in item):
            chinese_list.append(item)
        else:
            english_list.append(item)
    
    return chinese_list, english_list

def categorize_keywords(sorted_keywords_dict):
    chinese_keywords = {}
    english_keywords = {}
    
    for time, keywords in sorted_keywords_dict.items():
        for keyword in keywords:
            if any(char >= '\u4e00' and char <= '\u9fff' for char in keyword):
                if time not in chinese_keywords:
                    chinese_keywords[time] = []
                chinese_keywords[time].append(keyword)
            else:
                if time not in english_keywords:
                    english_keywords[time] = []
                english_keywords[time].append(keyword)
    
    return chinese_keywords, english_keywords

def remove_empty_time_entries(sorted_keywords_dict):
    return {time: keywords for time, keywords in sorted_keywords_dict.items() if time}

################################
# 关键词频率直方图
################################
def draw_bar(keyword_frequency, font_path, useremail, language):
    # 提取关键词和频次
    # 提取关键词和频次
    keywords = [item[0] for item in keyword_frequency][:40]
    frequencies = [item[1] for item in keyword_frequency][:40]

    # 设置字体
    font = FontProperties(fname=font_path, size=14)  # 将 "path_to_your_font_file.ttf" 替换为你的字体文件路径

    # 设置颜色映射
    color_map = cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min(frequencies), vmax=max(frequencies)))

    # 绘制直方图
    plt.figure(figsize=(20, 6))
    bars = plt.bar(keywords, frequencies, color=color_map.to_rgba(frequencies))
    plt.xlabel('Keyword', fontproperties=font)
    plt.ylabel('Frequency', fontproperties=font)
    plt.title('Keywords frequency bar', fontproperties=font)
    plt.xticks(rotation=45, ha='right', fontproperties=font)
    plt.tight_layout()

    # 添加颜色图例
    cbar = plt.colorbar(color_map, ax=plt.gca(), pad=0.01)
    # cbar.set_label('Frequency color', fontproperties=font)

    save_folder = os.getcwd() + '/literatureReview/' + useremail + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if language == "zh":
        save_path = save_folder + 'KeywordsFrequencyBar_zh.png'
    else:
        save_path = save_folder + 'KeywordsFrequencyBar_en.png'

    # 显示图形
    plt.tight_layout()  # 调整图形布局
    plt.savefig(save_path)

    return save_path

################################
# 关键词首次出现时间及随时间累积频率
################################
def time_key_to_key_time(md_files_info):

    # 转换为关键字-时间的列表，同时考虑小写相同和去掉s后相同的情况
    keyword_time_list = []
    for time, keywords in md_files_info.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_stripped = keyword_lower.rstrip("s")
            keyword_time_list.append((keyword_stripped, keyword_lower, time))

    # 按照关键字分组，并统计频次
    keyword_frequency = {}
    for stripped_keyword, lower_keyword, time in keyword_time_list:
        if stripped_keyword not in keyword_frequency:
            keyword_frequency[stripped_keyword] = {time: 1}
        else:
            if time not in keyword_frequency[stripped_keyword]:
                keyword_frequency[stripped_keyword][time] = 1
            else:
                keyword_frequency[stripped_keyword][time] += 1

    # 按照关键字进行排序
    sorted_keyword_frequency = sorted(keyword_frequency.items(), key=lambda x: x[0])
    result_dict = {}  # 初始化结果字典

    for item in sorted_keyword_frequency:
        keyword, time_dict = item
        result_dict[keyword] = time_dict
        
    time_line = list(md_files_info.keys())
        
    return result_dict, time_line

def draw_key_timeline(keyword_frequency, key_time_dict, timeline, font_path, useremail, language):
    font = FontProperties(fname=font_path, size=14)
    # 绘制折线图
    fig, ax = plt.subplots(figsize=(20, 7))
    # plt.figure(figsize=(20, 8))
    
    # 先画一个，要不刻度轴会乱
    zeros_list = []
    for i in range(len(timeline)):
        zeros_list.append(0)
    ax.plot(timeline, zeros_list)
    
    ano_pos = []
    for i in range(len(keyword_frequency)):
        ano_pos.append(i)

    keyword_index = 0
    key_time_frequency_dict = {}
    max_frequency = 0
    for keyword, _ in keyword_frequency:

        frequency_dict = key_time_dict[keyword.lower().rstrip("s")]
        frequencies = [frequency_dict.get(time, 0) for time in timeline]
        integral_list = [frequencies[0]]  # 初始化积分列表，第一个元素与原始列表一致

        for i in range(1, len(frequencies)):
            integral_list.append(integral_list[i - 1] + frequencies[i])
            
        frequencies = integral_list
        max_frequency = max(max_frequency, integral_list[-1])
            
        first_non_zero_index = next((index for index, value in enumerate(frequencies) if value != 0), None)
        #print(keyword)
        #print(list(frequency_dict.keys())[0])
        cur_date = list(frequency_dict.keys())[0]
        cur_fre = frequencies[first_non_zero_index]
        key_time_frequency_dict[keyword] = [cur_date, cur_fre]

        plt.plot(timeline[first_non_zero_index:], frequencies[first_non_zero_index:], label=keyword)
        
    #print(key_time_frequency_dict)
    sorted_key_time_frequency_dict = sorted(key_time_frequency_dict.items(), key=lambda x: x[1])
    #print(sorted_key_time_frequency_dict)
    for i in range(len(sorted_key_time_frequency_dict)):
        keyword = sorted_key_time_frequency_dict[i][0]
        cur_date = sorted_key_time_frequency_dict[i][1][0]
        cur_fre = sorted_key_time_frequency_dict[i][1][1]
        # # 标记坐标点
        ax.annotate(f'',
                    xy=(cur_date, cur_fre),
                    xytext=(cur_date, cur_fre + (i+1)*max_frequency/len(sorted_key_time_frequency_dict)),
                    arrowprops=dict(arrowstyle='->', linestyle='dotted', alpha=0.2, connectionstyle="angle"),
                   fontproperties=font)
        ax.annotate(f'{keyword} {cur_date}',
                    xy=(cur_date, cur_fre),
                    xytext=(cur_date, cur_fre + (i+1)*max_frequency/len(sorted_key_time_frequency_dict)),
                   fontproperties=font)

    plt.xlabel('Time', fontproperties=font)
    plt.ylabel('Frequency', fontproperties=font)
    plt.title('Temporal Cumulative Frequency of Keyword Occurrences', fontproperties=font)
    step = 5  # 每隔5个刻度
    selected_xticks = timeline[::step]  # 选择要显示的刻度位置
    plt.xticks(selected_xticks, rotation=45)

    legend = plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop=font)

    plt.tight_layout()

    save_folder = os.getcwd() + '/literatureReview/' + useremail + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if language == "zh":
        save_path = save_folder + 'KeywordsTimeline_zh.png'
    else:
        save_path = save_folder + 'KeywordsTimeline_en.png'

    plt.savefig(save_path)

    return save_path, sorted_key_time_frequency_dict

################################
# chatgpt
################################
def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an excellent researcher and paper writer."},
            {"role": "user", "content": prompt},
        ]
    )
    res = response["choices"][0]["message"]["content"]
    return res



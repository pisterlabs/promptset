"""
@Project ：list_ZH_CN
@File    ：settings
@Describe：
@Author  ：KlNon
@Date    ：2023/7/8 15:26
"""
import configparser
import json

import openai

# 引入配置文件
config = configparser.ConfigParser()
config.read('translate/config.ini', encoding='utf-8-sig')

"""
翻译前的准备工作相关配置
"""
# .jar文件所在的目录
JAR_DIR = config.get('START', 'jar_dir')
# 提取的文件夹要放置的目录
OUTPUT_DIR = config.get('START', 'output_dir')

COMPARE_ASSETS_ONE = config.get('START', 'compare_assets_one')
COMPARE_ASSETS_TWO = config.get('START', 'compare_assets_two')
COMPARE_ASSETS_THREE = config.get('START', 'compare_assets_three')

"""
翻译时的相关配置
"""
# 是否使用OpenAI
USE_OPENAI = config.getboolean('MAIN', 'use_openai')
# 是否手动校准
NEED_MANUAL_CONTROL = config.getboolean('MAIN', 'use_hand')
NEED_MANUAL_CONTROL_GROUPS = config.getboolean('MAIN', 'use_hand_word_groups')

# 你的OpenAI API密钥
openai.api_key = config.get('MAIN', 'openai_api_key')

# 中英对照表的路径
TRANSLATION_TABLE_WORDS_PATH = config.get('MAIN', 'translation_table_path_words')
TRANSLATION_TABLE_WORD_GROUPS_PATH = config.get('MAIN', 'translation_table_path_word_groups')
GPT_WORD_GROUPS_PATH = config.get('MAIN', 'gpt_path_word_groups')
SUCCESS_TRANSLATED_PATH = config.get('MAIN', 'success_translated_path')

DEL_PATH = config.get('MAIN', 'del_path')

# 读取中英对照表
with open(TRANSLATION_TABLE_WORDS_PATH, 'r', encoding='utf-8-sig') as f:
    translation_table_words = json.load(f)

with open(TRANSLATION_TABLE_WORD_GROUPS_PATH, 'r', encoding='utf-8-sig') as f:
    translation_table_word_groups = json.load(f)

with open(GPT_WORD_GROUPS_PATH, 'r', encoding='utf-8-sig') as f:
    gpt_word_groups = json.load(f)

with open(SUCCESS_TRANSLATED_PATH, 'r', encoding='utf-8-sig') as f:
    success_translated = json.load(f)


def refresh_json():
    global gpt_word_groups,success_translated
    with open(GPT_WORD_GROUPS_PATH, 'r', encoding='utf-8-sig') as f:
        gpt_word_groups = json.load(f)

    with open(SUCCESS_TRANSLATED_PATH, 'r', encoding='utf-8-sig') as f:
        success_translated = json.load(f)

import random
import time
import jsonlines
import json
import tiktoken
import openai
import os
from SeleniumInit import SeleniumInit
from lxml import etree
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from get_content import GetInfo
from svm_model import text_filter
from transformers import AutoTokenizer
from tqdm import tqdm
from utls.selenium_tool import selenium_entity
from info_extract import top_content_clean
from web_page_preprocess import noisy_text_clean


class BaiDuTop():
    """百度top5搜索结果返回"""
    def __init__(self, expert) -> None:
        self.expert = expert
        
        # self.url = f"https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&tn=baidu&wd={expert['name']}%20{expert['institute']}&oq=%25E8%2582%2596%25E6%2581%2592%25E4%25BE%25A8%2520%25E6%25B5%2599%25E6%25B1%259F%25E7%259C%2581%25E5%258C%2596%25E5%25B7%25A5%25E7%25A0%2594%25E7%25A9%25B6%25E9%2599%25A2%25E6%259C%2589%25E9%2599%2590%25E5%2585%25AC%25E5%258F%25B8&rsv_pq=f834336f00015541&rsv_t=8d70cS7osmXXImOBws0Bhy9AbGS5Shi%2FBnH3TYydqM2cEUJW7%2Fj0OfmzlJc&rqlang=cn&rsv_dl=tb&rsv_enter=0&rsv_btype=t" if expert else ''
        self.url = 'https://www.baidu.com/'
        self.selInit = selenium_entity()
        self.selInit.browser_run(url=self.url)
        self.time_sleep(1, 1.5)
        input_el = self.selInit.browser.find_elements(By.XPATH, '//input[@id="kw"]')[0]
        input_el.send_keys(f"{expert['name']}  {expert['scholar_institute']}")
        self.time_sleep(1.5, 2)
        # input_el.send_keys(Keys.ENTER)
        search_el = self.selInit.browser.find_elements(By.XPATH, '//input[@id="su"]')[0]
        search_el.click()
        self.time_sleep(2, 2.5)
    
    def time_sleep(self, a=1, b=3):
        """等待时间"""
        time.sleep(random.uniform(a, b))

    def get_el_prop_value(self, el, prop="innerHTML"):
        """获取标签属性内容"""
        try:
            _el = el
            if "list" in str(type(el)):
                _el = el[0]
            return etree.HTML(_el.get_attribute(prop)).xpath('string(.)')
        except Exception as e:
            return None
    
    def get_el(self, cover, xpath):
        """获取元素"""
        try:
            return cover.find_elements(By.XPATH, xpath)
        except Exception as e:
            return None
        
    def change_window_handle(self, selInit):
        """切换窗口句柄
        """
        handles = selInit.browser.window_handles  # 获取当前浏览器的所有窗口句柄
        selInit.browser.switch_to.window(handles[-1])  # 切换到最新打开的窗口

    def getInfoTop(self, top):
        """top限制最大为10"""
        if top > 10:
            top = 10
        # self.selInit.page_parse(url=self.selInit.browser.current_url)
        
        title_xpath = '//h3[contains(@class, "c-title")]/a'
        # 获取title元素
        title_el_list = self.get_el(cover=self.selInit.browser, xpath=title_xpath)
        url_list = []
        for title_el in title_el_list[:top]:
            try:
                title = self.get_el_prop_value(el=title_el)
                url = self.get_el_prop_value(el=title_el, prop="href")
                print("虚假url：", url)
                title_el.click()
                self.time_sleep(1,2)
            except Exception as e:
                print('标题元素获取异常:\n', str(e))
                continue
            try:
                self.change_window_handle(selInit=self.selInit)
            except Exception as e:
                print('浏览器窗口切换异常:\n',str(e))
                continue

            real_url = ""
            wait = True
            cnt = 0
            while wait and cnt < 20:
                try:
                    real_url = self.selInit.browser.current_url
                    print("真实url：", real_url)
                    wait = False
                except Exception as e:
                    print('当前页面url获取异常:\n', str(e))
                    cnt += 1
                    time.sleep(1)
            
            self.selInit.browser.close()     # 关闭当前窗口
            self.change_window_handle(selInit=self.selInit)
            if real_url != '':
                url_list.append({"title":title, "url":real_url})
        self.selInit.browser_close()
        return url_list

def main_info_check(experts_list):
    for expert in experts_list:
        main_info = 0
        
        check = lambda x: 0 if x == [] or x == None else 1

        if 'graduate_university' in expert:
            main_info |= check(expert['graduate_university'])
        if 'scholar_history' in expert:
            main_info |= check(expert['scholar_history'])
        if 'scholar_brief_info' in expert:
            main_info |= check(expert['scholar_brief_info'])
        if 'major_achievement_list' in expert:
            main_info |= check(expert['major_achievement_list'])
        
        expert['main_info'] = main_info
    
    return experts_list

def run():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # 测试数据入口
    experts_list = random.sample(list(jsonlines.open('./Data/res_04_28.jsonl')), 10)

    experts_list = main_info_check(experts_list)
    # experts_list = [expert for expert in main_info_check(experts_list) if expert['main_info'] == 0]
    # print(len(experts_list))
    for expert in tqdm(experts_list):
        print('\n当前处理专家:\n', expert)
        if expert['main_info'] == 1:
            with open('./Data/gpt_extract_res_test_ner_v1.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(expert, ensure_ascii=False) + '\n')
            continue
        start = time.time()
        wait = True
        cnt = 0
        while wait and cnt < 10:
            try:
                info_list = BaiDuTop(expert=expert).getInfoTop(top=5)
                wait = False
            except Exception as e:
                print('网络连接异常:\n', str(e))
                cnt += 1
                time.sleep(1)
        if cnt >= 10:
            print('当前处理网络请求超时重试次数达到上限...')
            continue
        index, tokens_count = 0, 0

        print('\n当前专家网页文本开始过滤...') 
        extract_info_list = []
        for info in info_list:
            index += 1
            info["content"] = GetInfo(info["url"]).get_page_content()
            info["expert_name"] = expert["name"]

            if isinstance(info["content"], str) and not info["expert_name"] in info["content"]:
                print("当前网页内容文本不包含目标专家实体，跳过处理...")
                continue

            print('\n当前专家分网页预处理开始...')
            if "AllBulletinDetail" in info["url"]:
                continue
            info["content"] = noisy_text_clean(info["content"], info["expert_name"]) # 网页内容预处理
            # print(info["content"])

            filtered = text_filter(info)
            token_ids = tokenizer.encode(filtered)
            tokens_count = len(tokenizer.encode(filtered, truncation=False))
            # 生成日志数据
            data_in = {
                'id': expert['id'],
                'name': expert['name'], 
                'institute': expert['scholar_institute'], 
                'filtered_content': filtered, 
                'tokens': tokens_count,
                'page_index': index,
                'url': info['url'],
                'main_info': expert['main_info']
            }
            extract_info_list.append(data_in)

            with open('./Data/cleaned_text_res1000_ner_v1.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_in, ensure_ascii=False))
                f.write('\n')
            # with jsonlines.open("./Data/origin_text_res1000.jsonl", 'a') as f:
            #     f.write(info)

        print('\n当前专家开始gpt信息抽取...')
        extract_res_list = top_content_clean(extract_info_list)
        for extract_res in extract_res_list:

            if '当前任职' in extract_res and extract_res['当前任职'] != 'unk':
                if 'occupation' in expert and isinstance(expert['occupation'], list):
                    expert['occupation'].append({
                        'content': extract_res['当前任职'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    })
                else:
                    expert['occupation'] = [{
                        'content': extract_res['当前任职'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    }]

            if '教育工作履历' in extract_res and extract_res['教育工作履历'] != 'unk':
                if 'scholar_history' in expert and isinstance(expert['scholar_history'], list):
                    expert['scholar_history'].append({
                        'content': extract_res['教育工作履历'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    })
                else:
                    expert['scholar_history'] = [{
                        'content': extract_res['教育工作履历'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    }]
                if 'scholar_history_source' in expert and isinstance(expert['scholar_history_source'], list):
                    expert['scholar_history_source'].append(extract_res['url'])
                else:
                    expert['scholar_history_source'] = [extract_res['url']]
                
            if '个人简介' in extract_res and extract_res['个人简介'] != 'unk':
                if 'scholar_brief_info' in expert and isinstance(expert['scholar_brief_info'], list):
                    expert['scholar_brief_info'].append({
                        'content': extract_res['个人简介'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    })
                else:
                    expert['scholar_brief_info'] = [{
                        'content': extract_res['个人简介'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    }]
                if 'scholar_brief_info_source' in expert and isinstance(expert['scholar_brief_info_source'], list):
                    expert['scholar_brief_info_source'].append(extract_res['url'])
                else:
                    expert['scholar_brief_info_source'] = [extract_res['url']]

            if '奖项成就' in extract_res and extract_res['奖项成就'] != 'unk':
                if 'major_achievement_list' in expert and isinstance(expert['major_achievement_list'], list):
                    expert['major_achievement_list'].append({
                        'content': extract_res['奖项成就'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    })
                else:
                    expert['major_achievement_list'] = [{
                        'content': extract_res['奖项成就'],
                        'url': extract_res['url'],
                        'tag': 'gpt-3.5-turbo'
                    }]
                if 'major_achievement_list_source'in expert and isinstance(expert['major_achievement_list_source'], list):
                    expert['major_achievement_list_source'].append(extract_res['url'])
                else:
                    expert['major_achievement_list_source'] = [extract_res['url']]
                    

        end = time.time()
        print("\n处理耗时: {:.2f} 秒".format(end - start))

        with open('./Data/gpt_extract_res_ner_v1.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(expert, ensure_ascii=False) + '\n')
            

if __name__ == '__main__':
    run()

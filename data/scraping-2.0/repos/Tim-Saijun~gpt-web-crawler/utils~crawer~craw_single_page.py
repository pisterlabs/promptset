import time
import requests
from bs4 import BeautifulSoup
import json
import re
import logging
from .openai_extract import openai_extract
from .get_screenshot import get_screenshot
def replace_multiple_spaces_with_single(s):
    return re.sub(r'\s+', ' ', s)

def check(url):
    try:
        base_domain = url.split('/')[2]
    except:
        return None
    if url.startswith('https://'+base_domain) or url.startswith('http://'+base_domain):
        return url
    # if url.startswith('/'):
    #     return 'https://'+base_domain+url
    else:
        return None

def craw_single_page(url,use_template=False,use_ai=False):
    url = check(url)
    if not url:
        return [],[],0
    response = requests.get(url)
    if response.status_code == 200:
        # 解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup.prettify())

        title = soup.title.string if soup.title else "N/A"
        # print(title)

        # 提取关键词
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            keywords = keywords['content']
        else:
            keywords = "N/A"

        description = soup.find('meta', attrs={'name': 'description'})
        if description:
            description = replace_multiple_spaces_with_single(description['content'])
        else:
            description = "N/A"

        # print("关键词：", keywords)
        # print("简介：", description)

        # for content_text in soup.find_all('meta', content=True):
        #     print(content_text.get('content'))
        if use_template: # 使用焦点建站的模板
            body_area = soup.find('div', attrs={'id': 'backstage-bodyArea'})
            body_content = body_area.get_text() if body_area else "N/A"
            body_content = replace_multiple_spaces_with_single(body_content)
            # print(body_content)
        else: # 一般的网站
            body_content = soup.body.get_text() if soup.body else "N/A"
            body_content = replace_multiple_spaces_with_single(body_content)
            # print(body_content)
        if use_ai:
        # 使用AI处理
            token_usage = 0
            meta_info = f"标题：{title}，关键词：{keywords}，简介：{description}"
            ai_extract_content,t = openai_extract(meta_info,body_content)
            token_usage += t
            # 判断json是否规范
            status = True
            time_out = 0
            while status and time_out<5:
                try:
                    json.dumps(ai_extract_content)
                    status = False
                except:
                    ai_extract_content,t = openai_extract(meta_info,body_content)
                    token_usage += t
                    time_out += 1
            
                
            print(ai_extract_content)
            print("🍎"*50)
        else:
            ai_extract_content = {}
            token_usage = 0
        
        # 提取网页截图
        # TODO: 多线程处理
        try:
            screenshot_path = get_screenshot(url)
        except Exception as e:
            logging.error(f"Error taking screenshot: {url}\n {e}")
            screenshot_path = None        
        logging.info(f"Screenshot saved to: {screenshot_path}")
        # 结算
        page_res = {"title": title,"url":url, "keywords": keywords, "description": description, "body": body_content,"ai_extract_content":ai_extract_content,"screenshot_path":screenshot_path}
        json_str = json.dumps(page_res, ensure_ascii=False)
        return page_res,json_str,token_usage

    else:
        print(url+"请求失败，状态码：", response.status_code)
        return {},{},0


if __name__ == '__main__':
    url = "https://www.jiecang.cn/220513140440.html"
    res = craw_single_page(url)[0]
    print(res)



import re
import json
import jieba
import openai
import random
import requests
import threading
import urllib.request
import urllib.parse
from tqdm import tqdm
from threading import Lock, Thread
from translate import Translator
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

openapipool=[]

def searchweibo(name):
    option = ChromeOptions()
    option.add_argument("--headless")
    browser = webdriver.Chrome(options=option)
    browser.get('https://weibo.com/login.php')
    with open('weibo_cookies.json', 'r', encoding='utf8') as f:
        listCookies = json.loads(f.read())
    for cookie in listCookies:
        cookie_dict = {
            'domain': '.weibo.com',
            'name': cookie.get('name'),
            'value': cookie.get('value'),
            "expires": '',
            'path': '/',
            'httpOnly': False,
            'HostOnly': False,
            'Secure': False
        }
    browser.add_cookie(cookie_dict)
    sleep(1)
    browser.get('https://weibo.com/login.php')
    WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "woo-pop-ctrl")))
    searchinput = browser.find_element(By.CLASS_NAME, 'woo-input-main')
    searchinput.send_keys(name)
    sleep(0.2)
    searchinput.send_keys(Keys.ENTER)
    new_window = browser.window_handles[-1]
    browser.close()
    browser.switch_to.window(new_window)
    WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "card-wrap")))
    weibo = browser.find_element(By.XPATH, '//div[@class="card card-user-b s-brt1 card-user-b-padding"]/div/a')
    weibo.click()
    new_window = browser.window_handles[-1]
    browser.switch_to.window(new_window)
    current_url = browser.current_url
    browser.get(current_url)
    response = browser.page_source
    browser.close()
    fans = re.findall("粉丝<span>(.+?)</span>", response)[0]
    print(current_url,fans)
    return current_url,fans

def chat_api(instr):
    # print("~~~ In ChatGPT ~~~")
    global openapipool
    try:
        openai.api_key =random.choice(openapipool)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instr},
            ],
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
    except:
        pass

def ChatTranslate(title):
    return chat_api("Translate it into English:"+title)

def generate_keywords(query):
    with open("/Users/bernoulli_hermes/projects/python_summer/query_format.md","r") as f:
        prompt=f.read().strip()
    prompt=prompt.replace("===text===",str(query[0:2000]))
    response=chat_api(prompt)
    try:
        return re.findall(r'\'\'\'\n(.+)\n\'\'\'',response,re.DOTALL)[0].split(", ")
    except:
        return []

def dlimg(url,imgname):
    try:
        response = requests.get(url)
    except:
        url="https://source.unsplash.com/1280x720/?tech"
        response = requests.get(url)
    img = response.content
    location="/Users/bernoulli_hermes/projects/python_summer/imgs/"+imgname+".png"
    with open(location, 'wb') as f:
        f.write(img)

def extract_chinese_words(text):
    chinese_pattern = re.compile('[\u4e00-\u9fa50-9]+')
    chinese_words = chinese_pattern.findall(text)
    return chinese_words

def refpage(i):
    global authors
    global refpages
    global pages
    aaaaa=[refpage["index"] for refpage in refpages]
    
    for page in tqdm(pages[15000+100*i:15000+100*i+100]):
        if page["index"] in aaaaa:
            continue
        refpage=page.copy()
        
        numimg=0
        textcontent=[]
        newcontent=[]
        
        if refpage["author_fans"]!="":
            pass
        else:
            if refpage["author"] in authors:
                refpage["author"]=authors[refpage["author"]]
            else:
                refpage["author_fans"]=0
                authors[refpage["author"]]=0
        
        for content in page['content']:
            if "https://n.sinaimg.cn/" in content:
                newcontent.append(content)
                numimg+=1
                try:
                    dlimg(content,str(page["index"])+"_"+str(numimg))
                except:
                    pass
            else:
                if "<" in content:
                    try:
                        newcontent.append(re.findall(r'>(.+?)<', content)[0])
                        textcontent.append(re.findall(r'>(.+?)<', content)[0])
                    except:
                        pass
                else:
                    newcontent.append(content)
                    textcontent.append(content)
        refpage["textcontent"]=textcontent
        refpage["newcontent"]=newcontent
        if numimg==0:
            # try:
            #     TitleinEnglish=Translator(from_lang="Chinese",to_lang="English").translate(page["title"])
            # except:
            TitleinEnglish=ChatTranslate(page["title"])
            try:
                dlimg("https://source.unsplash.com/1280x720/?"+TitleinEnglish,f'{refpage["index"]}_1')
            except:
                dlimg("https://source.unsplash.com/1280x720/?tech",f'{refpage["index"]}_1')
        
        seg_list = jieba.cut_for_search(''.join(extract_chinese_words("".join(refpage["textcontent"]))))
        seg_dict={}
        for seg in seg_list:
            if seg in seg_dict:
                seg_dict[seg]+=1
            else:
                seg_dict[seg]=1
        refpage["segdict"]=seg_dict
        
        if len(refpage["keywords"])<=5:
            refpage["keywords"]+=generate_keywords(''.join(refpage["newcontent"]))
        
        # refpages.append(refpage)

        with open("/Users/bernoulli_hermes/projects/python_summer/refined_pages1.json", "a") as f:
            f.write(","+json.dumps(refpage, indent=4, ensure_ascii=False))

pages=[]
authors={}
refpages=[]

def main():
    global pages
    global authors
    global refpages
    
    with open("/Users/bernoulli_hermes/projects/python_summer/sina_pages.json","r") as f:
        pages=json.loads(f.read())
    
    with open("/Users/bernoulli_hermes/projects/python_summer/refined_pages.json","r") as f:
        refpages=json.loads(f.read())

    for page in pages:
        if page["author_fans"]!="" and page["author"] not in authors:
            if "万" in page["author_fans"]:
                authors[page["author"]]=float(page["author_fans"].strip("万"))*10000
            else:
                authors[page["author"]]=float(page["author_fans"])


    for i in range(50):
        t = threading.Thread(target=refpage, args=(i,))
        t.start()

if __name__ == "__main__":
    main()
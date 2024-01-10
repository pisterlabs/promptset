from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime
import os
import openai
import time

openai.api_key = "xxxxx"

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d")

def get_url_text(url):
    driver.get(url)

    # 使用WebDriverWait等待特定元素加载完成
    # 这里以等待页面上的一个元素为例
    # wait = WebDriverWait(driver, 10)
    # element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "title")))

    # # 获取渲染后的页面内容
    # rendered_html = driver.page_source

    # # 在这里可以使用BeautifulSoup解析渲染后的HTML内容，提取所需的信息
    # # 这里只是一个示例，您可以根据实际需要进行修改
    # print("Rendered HTML:")
    # print(rendered_html)

    wait = WebDriverWait(driver, 10)
    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.content-inner')))

    # 保存页面内容到txt上：
    res = element.text
    # print("Element Text:", element.text)

    return res

def get_openai_content(input_data):
    messages = [
        {"role": "system", "content": "在1000字内用中文总结概括文章内容。"},
    ]

    message = input_data
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

    reply = chat.choices[0].message.content
    return reply

if __name__ == "__main__":
    """
    1.创建文件夹 
    """
    folder_path = 'data/%s' % formatted_time
    # 判定是否存在文件夹，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created")

    """
    2.爬内容
    """
    # 输入URL
    url_file = "./data/url.txt"
    url_list = []
    with open(url_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            url_list.append(line)

    # 配置Chrome WebDriver的选项
    options = Options()
    options.add_argument("--headless")  # 以无头模式运行，即不显示浏览器窗口
    # 添加不加载图片设置，提升速度
    # options.add_argument('blink-settings=imagesEnabled=false')
    # options.add_argument("interactive")
    driver = webdriver.Chrome(options=options)

    # 发送HTTP GET请求并获取渲染后的页面内容
    save_file_html_list = []
    for url in url_list:
        # url = "https://dataconomy.com/2023/07/19/meta-ai-what-is-llama-2-and-how-to-use/"
        # url = "https://dataconomy.com/2023/07/19/how-to-use-wormgpt-ai/"

        save_file_html = os.path.join(folder_path, url.split("/")[-2])
        save_file_html_list.append(save_file_html)

        res = get_url_text(url)
        time.sleep(3)

        with open(save_file_html, "w", encoding="utf-8") as f:
            f.write(res)

        print("Get %s content" % url)



    # 关闭WebDriver
    driver.quit()


    ''' 3.利用chatgpt进行要点概括 '''
    save_chatgpt_file = open(os.path.join(folder_path, "%s.txt" % formatted_time), "w", encoding="utf-8")
    for save_file_html, url in zip(save_file_html_list, url_list):
        with open(save_file_html, "r", encoding="utf-8") as f:
            res = f.read()

        reply = get_openai_content(res)
        print(reply)
        print(("#")*15)

        save_chatgpt_file.write(" ".join(save_file_html.split("/")[-1].split("-")) + "\n")
        save_chatgpt_file.write(reply + "\n")
        save_chatgpt_file.write("文章链接：" + "[%s](%s)" % (url, url) + "\n")
        save_chatgpt_file.write("\n")

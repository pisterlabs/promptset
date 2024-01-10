import openai
import time
import threading

openai.api_key = ""
def translate(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"完整翻譯成繁體中文.: {text}",
        temperature=0.0,
        max_tokens=200,
        stop="\t"
    )
    translation = response.choices[0].text.strip()
    return translation

# 处理单行文本的函数
def translate_line(index, line, shared_data):
    translation = translate(line)
    with open('ch3.txt', 'a', encoding='utf-8') as f:
        shared_data[index] = translation


# 讀取檔案
def translate_text(shared_data):
    # 创建线程列表
    threads = []
    
    for index, line in enumerate(shared_data):
        thread = threading.Thread(target=translate_line, args=(index, line, shared_data))
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    

def main():
    start_time = time.time()
    file_path = "jp.txt"
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    shared_data = [None] * len(lines)
    for i, line in enumerate(lines):
        shared_data[i] = line[:-1]

    translate_text(shared_data)
    
    for data in shared_data:
        print(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"總共耗時：{elapsed_time} 秒")

if __name__ == "__main__":
    main()
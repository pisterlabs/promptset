import openai
import time
import threading

class LineReadThread(threading.Thread):
    def __init__(self, index, line, shared_data):
        threading.Thread.__init__(self)
        self.index = index
        self.line = line
        self.shared_data = shared_data

    def run(self):
        translation = translate(self.line)
        self.shared_data[self.index] = self.translation

    def translate(self):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"完整翻譯成繁體中文.: {self.line}",
            temperature=0.0,
            max_tokens=200,
            stop="\t"
        )
        translation = response.choices[0].text.strip()
        return translation

openai.api_key = ""


def main():
    start_time = time.time()
    file_path = "jp.txt"
    # 讀取jp.txt
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    shared_data = [None] * len(lines)
    for i, line in enumerate(lines):
        shared_data[i] = line[:-1]

    threads = []
    
    for index, line in enumerate(shared_data):
        thread = LineReadThread(index, line, shared_data)
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    for data in shared_data:
        with open('ch4.txt', 'a', encoding='utf-8') as f:
            f.write(data + '\n')


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"總共耗時：{elapsed_time} 秒")

if __name__ == "__main__":
    main()
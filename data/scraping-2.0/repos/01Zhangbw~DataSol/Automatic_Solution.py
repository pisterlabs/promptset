import threading
import queue
import os
import openai
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 定义你的四个OpenAI API密钥
OPENAI_API_KEYS = [
    "YOUR_API_KEY_1",
    "YOUR_API_KEY_2",
    "YOUR_API_KEY_3",
    "YOUR_API_KEY_4"
]

def chat4(prompt, temperature=0.7):
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    retry_count = 0  # 重试计数器
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",  # 使用所需的模型
                messages=messages,
                temperature=temperature
            )
            answer = completion.choices[0].message['content']
            return answer
        except Exception as e:
            if retry_count >= 5:  # 连续失败5次后跳过此提示
                print(f"Failed to process prompt after 5 retries: {prompt}")
                return None
            retry_count += 1  # 增加重试计数器

# 定义要使用的线程数量
NUM_THREADS = len(OPENAI_API_KEYS)

# 创建线程安全的任务队列和结果队列
task_queue = queue.Queue()
result_queue = queue.Queue()

# 定义子文件夹的目录路径
SUBFOLDER_PATHS = ["0", "1", "2", "3"]

# 定义用于存储文件夹优先级的字典
folder_priorities = {folder: int(folder) for folder in SUBFOLDER_PATHS}

# 创建一个集合以跟踪已处理的文件
processed_files = set()

# 处理文件的函数
def process_file(file_path, api_key):
    openai.api_key = api_key
    with open(file_path, 'r') as file:
        content = file.read()

    # 在这里执行GPT-3.5 API处理
    response = chat4(content)  # 你应该定义自己的chat4函数来进行API调用

    # 保存响应或进行进一步处理
    # ...

    print(f"Processed file: {file_path} with API key: {api_key}")

# 处理文件夹中的文件的函数
def process_folder(api_key):
    while True:
        file_path = task_queue.get()
        if file_path is None:
            break

        folder_priority = folder_priorities[api_key]
        current_file_priority = folder_priorities[file_path]

        if current_file_priority >= folder_priority:
            process_file(file_path, api_key)
        else:
            print(f"Skipping lower priority file: {file_path}")

        task_queue.task_done()

# 检查并处理文件夹中的文件的函数
def check_and_process_folders():
    while True:
        for folder_path in SUBFOLDER_PATHS:
            files = os.listdir(folder_path)
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                if file_path not in processed_files:
                    task_queue.put(file_path)
                    processed_files.add(file_path)

        time.sleep(1)  # 每隔一秒检查新文件

# 处理新添加到文件夹的文件的函数
def handle_new_file(file_path):
    print(f"New file detected: {file_path}")
    # 将文件添加到处理队列或采取适当的措施
    task_queue.put(file_path)

# 启动文件夹监控的函数
def start_folder_monitoring():
    observer = Observer()
    event_handler = FileSystemEventHandler()

    for folder_path in SUBFOLDER_PATHS:
        observer.schedule(event_handler, folder_path, recursive=False)

    event_handler.on_created = handle_new_file

    observer.start()

try:
    # 启动文件夹监控
    start_folder_monitoring()

    # 创建并启动工作线程
    worker_threads = []
    for api_key in OPENAI_API_KEYS:
        worker_thread = threading.Thread(target=process_folder, args=(api_key,))
        worker_thread.start()
        worker_threads.append(worker_thread)

    # 创建一个线程来检查并处理文件夹
    folder_thread = threading.Thread(target=check_and_process_folders)
    folder_thread.start()

    # 主循环，用于处理用户交互或其他任务
    while True:
        user_input = input("Enter 'quit' to exit: ")
        if user_input == 'quit':
            # 通知工作线程退出
            for _ in range(NUM_THREADS):
                task_queue.put(None)
            break

    # 等待所有工作线程完成
    for worker_thread in worker_threads:
        worker_thread.join()

    # 等待文件夹线程完成
    folder_thread.join()

    print("Main program has finished.")
except KeyboardInterrupt:
    print("Program terminated by user.")

import queue
import threading
import openai

# 初始化隊列和事件
ChatQueue = queue.Queue()
Finished = threading.Event()
StopEvent = threading.Event()
ChatContentTimer = None

def start_timer():
    """
    開始計時器
    """
    global ChatContentTimer
    ChatContentTimer = threading.Timer(5.0, timeout_callback)
    ChatContentTimer.start()

def reset_timer():
    """
    重置計時器
    """
    global ChatContentTimer
    ChatContentTimer.cancel()
    start_timer()

def timeout_callback():
    """
    計時器回調函數
    """
    global ChatQueue
    global ChatContentTimer

    # 重置計時器時間為0
    ChatContentTimer = None

    # 等待現有聊天請求被處理完畢
    while not ChatQueue.empty():
        pass

    # 結束隊列處理循環
    StopEvent.set()

def chat_worker():
    """
    聊天請求處理工作線程
    """
    global ChatQueue
    global Finished
    global StopEvent

    start_timer()

    while not StopEvent.is_set():
        try:
            # 從隊列中取出聊天請求
            chat_content = ChatQueue.get(timeout=1)

            # 使用OpenAI API進行聊天
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens = 80,
                temperature = 1,
                messages=[
                    {"role": "system", "content": "你將會把使用者給予的文本翻譯成英文。"},
                    {"role": "user", "content": chat_content}
                ]
            )

            # 返回處理結果
            result = completion.choices[0].message.content
            print(result)

            # 將處理結果加入 Finished 列表中
            Finished.put(result)

            # 重置計時器
            reset_timer()

            # 標記該請求已經完成
            ChatQueue.task_done()

        except queue.Empty:
            pass

def chat_request(chat_content):
    """
    將新的聊天請求添加到ChatQueue中，並開始計時器
    """
    global ChatQueue

    # 重置計時器
    if ChatContentTimer is not None:
        reset_timer()

    # 將新的聊天請求添加到ChatQueue中
    ChatQueue.put(chat_content)

    # 開始計時器
    start_timer()

if __name__ == '__main__':
    # 啟動多個聊天請求處理器
    num_workers = 1
    for i in range(num_workers):
        threading.Thread(target=chat_worker).start()

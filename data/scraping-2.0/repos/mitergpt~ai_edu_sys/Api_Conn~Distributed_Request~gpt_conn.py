import threading
import openai
import time
import random

api_key_1 = "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*13112dasdasd"
api_key_2 = "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*2%eqeqeqe"

lock_1 = threading.Lock()
lock_2 = threading.Lock()
thread_count_1,thread_count_2=0,0
thread_count=[thread_count_1,thread_count_2] #记录下每条路径的执行的进程数目的情况
time_list=[[],[]]

def make_request(server_i,api_key, message_1, message_2, thread_count, lock):
    start_time = time.time()
    cycle=3
    with lock:
        thread_count += 1
    openai.api_key = api_key
    while cycle>0 :
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": message_1},
                    {"role": "user", "content": message_2},
                ],
                temperature=0,
                max_tokens=1024
            )
        except:
            cycle -=1
            return
    with lock:
        thread_count -= 1
    end_time = time.time()
    wait_time=end_time-start_time #处理时间
    time_list[server_i].append(wait_time) #存入最近一段时间的处理数据集，
    return response

#对节点的时间进行统计，负载均衡
while True:
      total_1,total_2=0,0
      for a in  time_list[0]:
          total_1+=a
      for b in time_list[1]:
          total_2+=b
      time.sleep(1)
p=total_1/total_2+total_1

if random.random()>p :
   thread_1 = threading.Thread(target=make_request, args=(1,api_key_1, '我是刘洋', '你是喜羊羊',thread_count[0],lock_1))
else :
   thread_1 = threading.Thread(target=make_request, args=(2,api_key_2, '我是刘洋', '你是喜羊羊',thread_count[1],lock_2))

if random.random()>p :
   thread_2 = threading.Thread(target=make_request, args=(1,api_key_1, '我是刘洋', '你是喜羊羊',thread_count[0],lock_1))
else :
   thread_2=threading.Thread(target=make_request, args=(2,api_key_2, '我是刘洋', '你是喜羊羊',thread_count[1],lock_2))

thread_1.start()
thread_2.start()
thread_1.join()
thread_2.join()

"""后续需要把这些api进行扩展，还需要进一步考虑负载均衡，备份节点和运行速度，三者之间的平衡"""

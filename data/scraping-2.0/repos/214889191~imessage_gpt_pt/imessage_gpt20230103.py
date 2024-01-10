# --*-- conding:utf-8 --*--
# @Time : 2023/01/03 11:47
# @Author :pthaochiya
# @Email :
# @File : imessage_gpt2023.py
# @Software : PyCharm
# 使用 Snake Case（蛇形命名法） 来命名变量和函数（全部小写+下划线）my_variable_name
# 使用 Pascal Case（帕斯卡拼写法） 来命名类（首字母全部大写）myVariableName
# （一般不用）Camel Case（骆驼拼写法）：首字母小写，其余单词首字母大写（驼峰式命名法）MyClassName

"""
使用读：
重构imessage_gpt项目代码
1、监听自己的imessage消息
2、发现新消息后，调用gpt模型生成回复
3、将回复消息发送给对应的imessage用户
4、需要对消息进行保存，以实现上下文的记忆对话
5、使用本地csv文件保存消息，和提取消息
注意：
1、imessage的数据是存储在sqlite数据库中的，需要使用python的imessage库来读取
2、需要在本地安装imessage的python库
3、需要在本地安装openai的python库（如果openai的库有更新，需要自己更新代码，以适应变化）
4、为每一个用户创建一个csv文件，用于保存消息，发现新用户的时候才创建对应的文件（为了隐私，建议不要这个功能，把用户的对话消息放在内存里）
5、查找用户目录中，是否存在对应的csv文件，如果不存在则创建，存在则读取
6、本代码没有对用户的发送消息频率进行限制，如果需要，可以自己添加
"""

import os
import time
import threading
import csv
import sqlite3
import openai
import queue

# Python内置模块 subprocess的语句，可以用于在Python脚本中执行外部命令或程序
import subprocess
import openai

# 设置API key
print('设置API key：')
print('设置gpt模型的名称,根据你的token来选择：\n'
      'gpt-3.5-turbo:  速度快，但是回复的内容不太准确\n'
      'gpt-4-0314:     小模型 \n'
      'gpt-4-32k-0314: 大模型，是前者的token消耗是后者的4倍')
openai.api_key = input("请输入你的openai的API key>>>>")
# 设置你电脑的imessage的数据库路径（记得关闭苹果的系统保护，自行百度即可，这里不做赘述）
mmy_sqlite_path = os.path.expanduser("/Users/xxxx（比电脑的名字）/Library/Messages/chat.db")
# 设置你的imessage邮箱(最好注册一个新的，垃圾邮件少一点)
iMessage_id = "xxxxxxx@gmail.com"
# 设置上下文的消息数量，用于生成回复(数量越多，消耗越多的token，但是回复的内容也越准确)
context_num = -5
# 设置gpt模型的名称（这里写死的，按需修改）
gpt_model = "gpt-3.5-turbo",


# 发送消息功能 发文字、发图片都用的这个
def sending_imessage(recipient, chatgpt_message):
    # 接收号码可以是手机号，可以是imessage邮箱
    recipient = recipient
    # 要发送的内容
    message = chatgpt_message
    # 定义调用系统imessage发送消息的苹果脚本语言
    # 图片举例：'tell application "Messages" to send "/path/to/image.jpg" to buddy "Contact Name"'], check=True
    script = f'tell application "Messages" to send {{"{message!r}"}} to buddy "{recipient}"'
    cmd = ['osascript', '-e', script]
    # 执行发送
    subprocess.run(cmd)
    print("sending_imessage函数，消息发送成功，Message sent successfully.")


# 定义一个函数，用于获取用户的imessage信息，定时获取
def get_new_imessage():
    # 定义一个变量，用于保存最新消息的ROWID
    last_message_rowid = ''
    # 连接到imessage数据库
    sqlite_conn = sqlite3.connect(mmy_sqlite_path)
    while True:
        time.sleep(3)
        # 获取当前时间
        now_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

        # 定义查询语句，查询最新消息的ROWID，即发消息在数据库中的ID
        now_message_rowid_sql = "SELECT ROWID FROM message ORDER BY date DESC LIMIT 1"
        # 执行查询语句
        now_message_rowid = sqlite_conn.execute(now_message_rowid_sql).fetchone()[0]
        # print(f"{now_time}:执行查询，最新消息的ROWID是：{now_message_rowid}" )

        """
        # 定义查询语句，查询destination_caller_id的值，即发送人的电话号码
        # 判断消息的目的地是还不是imessage，如果是一般就是imessage消息，如果不是，则不处理,可能是营销短信等
        # 根据需要，可以将这个判断去掉
        destination_caller_id_sql = "SELECT destination_caller_id FROM message ORDER BY date DESC LIMIT 1"
        # 执行查询语句
        destination_caller_id = sqlite_conn.execute(destination_caller_id_sql).fetchone()[0]
        # 打印查询结果
        print("发现最新消息的目的地是：", destination_caller_id)
        if destination_caller_id != iMessage_id:
            print("发现最新消息的目的地不是iMessage，不处理！")
            pass
        """

        if last_message_rowid == '':
            print(f'第一次运行，不需要回复！')
            last_message_rowid = now_message_rowid
            # 跳过本次循环，继续下一次循环
            continue
        # 判断当前这条消息的时间，是否是在本次程序运行之后，如果是，则是新消息，需要回复
        # print(f'当前消息的ROWID是：{now_message_rowid},上一次消息的ROWID是：{last_message_rowid}')
        if int(now_message_rowid) > int(last_message_rowid):
            # 等待回复数量
            wait_reply_num = int(now_message_rowid) - int(last_message_rowid)
            print(f'发现新消息，需要回复！{wait_reply_num}条！')
            # 更新最新消息的ROWID
            last_message_rowid = now_message_rowid

            # 需要回复的消息ROWID分别是
            wait_reply_message_rowid = [int(last_message_rowid) - i for i in range(wait_reply_num)]
            print(f'需要回复的消息，ROWID分别是：{wait_reply_message_rowid}')

            for j in wait_reply_message_rowid:
                # 定义查询语句，通过消息的ROWID，查询该条消息的，发送人员的imessage号码
                imessage_sender_sql = "SELECT handle_id FROM message WHERE ROWID = ?"
                # 执行查询语句
                imessage_sender = sqlite_conn.execute(imessage_sender_sql, (j,)).fetchone()
                if imessage_sender:
                    sender_handle_id = imessage_sender[0]
                    print("消息发送人imessage的id是：", sender_handle_id)
                else:
                    print("未找到与指定ROWID相对应的消息")

                # 到imessage数据库的，人员列表中去查询当前这条消息，是谁发送的，获取到发件人的联系方式
                query = f'SELECT id FROM handle WHERE ROWID = {int(sender_handle_id)}'
                sender_information_num = sqlite_conn.execute(query).fetchone()[0]
                print(f'消息的发送人员的imessage号是:{sender_information_num}')

                # 定义查询语句，查询最新消息的内容
                query_message_sql = "SELECT text FROM message WHERE ROWID = ?"
                # 执行查询语句
                last_imessage_text = sqlite_conn.execute(query_message_sql, (j,)).fetchone()[0]
                # 打印查询结果
                print("发现最新消息的内容是：", last_imessage_text)
                # 判断数据是不是NOne，如果是，则不处理
                if last_imessage_text is None:
                    print("发现最新消息的内容是None，不处理！")
                    continue

                # 判断当前这条消息的发送人，是否是已经有了对应的csv文件，如果没有，则创建一个
                # 定义变量，保存当前用户的csv文件路径
                user_csv_pass = f'./{sender_information_num}.csv'
                # 判断当前用户的csv文件是否存在，如果不存在，则创建一个
                if not os.path.exists(user_csv_pass):
                    print(f'发现新用户，创建对应的csv文件:{user_csv_pass}')
                    # 创建一个csv文件，用于保存消息
                    with open(user_csv_pass, 'w', newline='') as f:
                        # 创建一个csv写入对象
                        csv_write = csv.writer(f)
                        # 写入一行数据
                        csv_write.writerow([now_time, sender_information_num + ':', last_imessage_text])

                else:
                    print(f'发现老用户，追加消息到csv文件:{user_csv_pass}')
                    # 如果存在，追加本次文件到csv文件中
                    with open(user_csv_pass, 'a', newline='') as f:
                        # 创建一个csv写入对象
                        csv_write = csv.writer(f)
                        # 写入一行数据
                        csv_write.writerow([now_time, sender_information_num + ':', last_imessage_text])

                # 更新最新消息的ROWID
                print("更新最新消息的ROWID")
                last_message_rowid = now_message_rowid

                # 将发送人员的imessage号码，放到字典中，用于传递给其他的线程
                message_dict = {'sender_information_num': sender_information_num}
                # 将消息放到消息队列里面，用其他的线程去处理
                message_queue.put(message_dict)

        else:
            print(f'{now_time}:没有发现新消息，不需要回复！等待下一次检查')


# 定义一个函数，用来调用gpt模型，生成回复消息
def handle_message(message_dict):
    # 把用户的联系方式提取出来
    sender_information_num = message_dict['sender_information_num']
    # 创建一个空列表，准备要发送的上下文数据 list列表数据类型
    send_to_gpt_text = []
    # 读取用户文件中的所有消息
    all_messages = []

    with open(f'./{sender_information_num}.csv', 'r', newline='') as f:
        csv_read = csv.reader(f)
        for row in csv_read:
            all_messages.append(row)

    # 输出最新5条消息，修改下面的数字，可以改变输出的历史消息数量
    latest_messages = all_messages[context_num:]
    # print("最新5条消息：")
    for message in latest_messages:
        print(message)
        # 将消息添加到上下文数据中，根据回复的内容区别是用户还是助手
        if message[1] == 'chatgpt:':
            send_to_gpt_text.append({"role": "assistant", "content": message[2]})
        else:
            send_to_gpt_text.append({"role": "user", "content": message[2]})
    print(f'即将发送，上下文数据：{send_to_gpt_text}')

    # 给ChatGPT发送请求，openai.ChatCompletion类用于对话用途
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=send_to_gpt_text,
        # 使用什么采样温度，介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使输出更加集中和确定。
        temperature=0.2,
        # 消耗的长度控制
        max_tokens=1024,
    )
    # 打印请求结果
    # print(completion.choices[0].message)
    gpt_answer_text = completion.choices[0].message.content
    print(f'chatgpt答复内容：{gpt_answer_text}')
    # 将回复消息保存到csv文件中
    # 定义变量，保存当前用户的csv文件路径  message_dict['sender_information_num']
    user_csv_pass = f'./{sender_information_num}.csv'
    # 如果存在，追加本次文件到csv文件中
    with open(user_csv_pass, 'a', newline='') as f:
        # 创建一个csv写入对象
        csv_write = csv.writer(f)
        # 写入一行数据
        csv_write.writerow([time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), 'chatgpt:', gpt_answer_text])

    # 发送消息给用户
    sending_imessage(message_dict['sender_information_num'], gpt_answer_text)


# 定义一个函数用来消费消息队列中的消息
def handle_message_queue():
    while True:
        # 从消息队列中获取消息，从队列中获取消息，并且在获取后，该消息会从队列中移除。
        message = message_queue.get()
        print(f'从消息队列中获取消息：{message}')
        # 处理消息
        handle_message(message)
        # 标记消息处理完成，用于通知队列，表示你已经处理完这个消息，队列可以将其从内部计数中删除
        message_queue.task_done()


if __name__ == '__main__':
    # 初始化一个空的消息队列
    message_queue = queue.Queue()
    print("程序初始化等待3秒！")

    # 创建一个线程，用于定时获取imessage消息
    t1 = threading.Thread(target=get_new_imessage)
    # 根据用户的使用量，可以增加线程的数量，修改range的值，就可以改变线程的数量
    for i in range(1):
        # 创建一个线程，用于处理消息队列中的消息
        t2 = threading.Thread(target=handle_message_queue)
        t2.start()

    # 启动线程
    t1.start()
    # 等待线程结束
    t1.join(timeout=3)
    t2.join(timeout=3)

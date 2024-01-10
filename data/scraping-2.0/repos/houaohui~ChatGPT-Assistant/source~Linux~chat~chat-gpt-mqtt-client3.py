from paho.mqtt import client as mqtt_client
import time
import openai
import requests
import json
import sys

topic_A = "/test/answer"
topic_Q = "/test/question"

#proxy = "127.0.0.1:10808"
#proxies = {
#  'http': f'socks5://{proxy}',
#  'https': f'socks5://{proxy}',
#}

messages=[{"role": "system", "content": "你是一个有用的助手"}]

def get_total_size(obj):
    size = sys.getsizeof(obj)
    if hasattr(obj, '__iter__'):
        if hasattr(obj, 'items'):
            for key, value in obj.items():
                size += get_total_size(key)
                size += get_total_size(value)
        elif not isinstance(obj, str):
            for item in obj:
                size += get_total_size(item)
    return size

# 连接mqtt服务器
def connect_mqtt():
    # 服务器信息
    broker = '127.0.0.1'
    port = 1883
    # 客户端id
    client_id = 'chat-gpt'

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# 订阅回调函数
def on_message(client, userdata, msg):
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    input_str = f"{msg.payload.decode()}"
    if input_str == "new chat":
        messages.clear()
        messages.append({"role": "system", "content": "你是一个有用的助手"})
        return
    messages.append({"role": "user", "content":input_str})
    text = chatWithAi(messages)
    if text == "error":
        publish(client, topic_A, "我好像出了点问题,再试一下")
        return
    text = text.replace("\n","")
    messages.append({"role": "assistant", "content":text})
    publish(client, topic_A, text)

# 订阅消息


def subscribe(client: mqtt_client, topic):
    client.subscribe(topic)
    client.on_message = on_message

# 发布消息


def publish(client, topic, msg):
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")




def chatWithAi(prompt):
    url = "https://api.openai.com/v1/chat/completions"


    data = {
        "messages": prompt,
        "model":"gpt-3.5-turbo",
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
        "n": 1
    }


    headers = {
        'Authorization': 'Bearer 你的api-key',
        'Content-Type': 'application/json'
    }

    response = requests.request(
        "POST", url, headers=headers, json=data, allow_redirects=False)
    json_data = response.json()


    
    # 格式化打印
    formatted_data = json.dumps(json_data, indent=4)
    print(formatted_data)


    
    # 自动消减请求长度
    total_tokens = json_data['usage']['total_tokens']
    print(f"total_tokens: {total_tokens}")
    size = get_total_size(prompt)
    print(f"messages size: {size}")
    last_size = size

    if total_tokens > 3500:
        while size > last_size/2 :
            del prompt[0]
            size = get_total_size(prompt)
        print(f"now size: {size}")

    # 检查字典中是否存在 error 键
    if "error" in json_data:
        return "error"

    result_txt = json_data['choices'][0]['message']['content']
    return result_txt



if __name__ == "__main__":

    client = connect_mqtt()
    client.loop_start()
    # 订阅问题
    subscribe(client, topic_Q)
    while (True):
        #input_str = input("you: ")
        #if str == 'q':
        #    break
        #messages.append({"role": "user", "content":input_str})
        #text = chatWithAi(messages)
        #text = text.replace("\n","")
        #print("AI: " + text)
        #messages.append({"role": "assistant", "content":text})
        # print(messages)

        time.sleep(1)

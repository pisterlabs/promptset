#!/usr/bin/env python3
# date:      2023/10/19

import socket,base64,hashlib
import openai

def get_headers(data):
    '''将请求头转换为字典'''
    header_dict = {}
    data = str(data,encoding="utf-8")

    header,body = data.split("\r\n\r\n",1)
    header_list = header.split("\r\n")
    for i in range(0,len(header_list)):
        if i == 0:
            if len(header_list[0].split(" ")) == 3:
                header_dict['method'],header_dict['url'],header_dict['protocol'] = header_list[0].split(" ")
        else:
            k,v=header_list[i].split(":",1)
            header_dict[k]=v.strip()
    return header_dict

def get_data(info):
    payload_len = info[1] & 127
    if payload_len == 126:
        extend_payload_len = info[2:4]
        mask = info[4:8]
        decoded = info[8:]
    elif payload_len == 127:
        extend_payload_len = info[2:10]
        mask = info[10:14]
        decoded = info[14:]
    else:
        extend_payload_len = None
        mask = info[2:6]
        decoded = info[6:]

    bytes_list = bytearray()    #这里我们使用字节将数据全部收集，再去字符串编码，这样不会导致中文乱码
    for i in range(len(decoded)):
        chunk = decoded[i] ^ mask[i % 4]    #解码方式
        bytes_list.append(chunk)
    body = str(bytes_list, encoding='utf-8')
    return body

def send_msg(conn, msg_bytes):
    """
    WebSocket服务端向客户端发送消息
    :param conn: 客户端连接到服务器端的socket对象,即： conn,address = socket.accept()
    :param msg_bytes: 向客户端发送的字节
    :return:
    """
    import struct

    token = b"\x81" #接收的第一字节，一般都是x81不变
    length = len(msg_bytes)
    if length < 126:
        token += struct.pack("B", length)
    elif length <= 0xFFFF:
        token += struct.pack("!BH", 126, length)
    else:
        token += struct.pack("!BQ", 127, length)

    msg = token + msg_bytes
    conn.send(msg)
    return True

def get_api_ask(que):
    open_key = "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"
    openai.api_key = open_key
    modelAi = "gpt-3.5-turbo"
    content = que
    response = openai.ChatCompletion.create(model=modelAi,messages=[{"role":"user","content": content}])
    #response = openai.Image.create(prompt="dog",n=1,size="102x102")
    #print(response)
    ask = response['choices'][0]['message']['content']
    return ask

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
sock.bind(("0.0.0.0",8080))
sock.listen(5)

#等待用户连接
conn,addr = sock.accept()
#获取握手消息，magic string ,sha1加密
#发送给客户端
#握手消息
data = conn.recv(8096)

headers = get_headers(data)

# 对请求头中的sec-websocket-key进行加密
response_tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
      "Upgrade:websocket\r\n" \
      "Connection: Upgrade\r\n" \
      "Sec-WebSocket-Accept: %s\r\n" \
      "WebSocket-Location: ws://%s%s\r\n\r\n"

magic_string = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

value = headers['Sec-WebSocket-Key'] + magic_string
ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())

response_str = response_tpl % (ac.decode('utf-8'), headers['Host'], headers['url'])

# 响应【握手】信息
conn.send(bytes(response_str, encoding='utf-8'))
#send_msg(conn,bytes(response_str,encoding="utf-8"))

#可以进行通信
while True:
    print("coming")
    mydata = conn.recv(8096)
    if mydata:
        data = mydata
        data = get_data(data)
        print(data)
        res = get_api_ask(data) 
        #send_msg(conn,bytes(data+"geah",encoding="utf-8"))
        send_msg(conn,bytes(res,encoding="utf-8"))
    else:
        break

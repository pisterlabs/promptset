import socket
import threading
import json
from firebase import firebase 
from face.facedetect import register,login
from face.face_identify import identify
from speech.gpt import openai_api
import gui_button.firebase_  as firebase_
from speech.reply import reply
import time

# 服务器配置
HOST = ''  # 服务器IP地址
PORT = 7100  # 服务器端口号

# 用于存储连接的客户端信息
clients = []
# 创建锁，用于同步访问clients列表
lock = threading.Lock()

def send_message_to_db(message):
    url = "https://python-database-3b3f8-default-rtdb.firebaseio.com/"
    fdb = firebase.FirebaseApplication(url, None)    # 初始化，第二個參數作用在負責使用者登入資訊，通常設定為 None
    fdb.put('/togui','now',message)

def handle_client(client_socket, client_address):
    dic = {"伏特加萊姆":16,"伏特加七喜":17,"柯夢波丹":24,"海風":18,"琴通寧":20,"龍舌蘭日出":22,"威士忌可樂":23,"長島冰茶":21,"性慾海灘":19}
    while True:
        try:
            data = client_socket.recv(1024)  # 接收客户端发送的数据
            if data:
                # 处理接收到的数据
                print(f"Received data from {client_address}: {data.decode()}")
                url = "https://python-database-3b3f8-default-rtdb.firebaseio.com/"
                fdb = firebase.FirebaseApplication(url, None)
                command=data.decode().split()
                if(command[0]=="startprogram"):
                    v=""
                    status = fdb.get('/arduino/座位',None)
                    print("status:",status)
                    if status==1:
                        v='1'
                        fdb.put('/','ppt',4)
                    else:
                        v='0'
                    users = fdb.get('/user_data',None)
                    for people,data in users.items():
                        fdb.put('/user_data/'+people,'nowlogin',0)
                    fdb.put('/','ppt',8)                       
                    #firebase sit 0 or 1 ,v=0,v=1         
                    with lock:
                    # 向所有其他客户端广播数据
                        for client in clients:
                            if client == client_socket:
                                my_dict = {"startprogram": v}  # create your dictionary here
                                json_data = json.dumps(my_dict)  # convert dictionary to json
                                json_data +='\n'
                                client.sendall(json_data.encode('utf-8'))  # send json data
                    #queue.put(f'{client_address} get in')
                    print(f'{client_address[0]} get in')
                    send_message_to_db(f'{client_address[0]} get in.')
                elif (command[0]=="register"):
                    fdb.put('/','ppt',5)
                    v = register(command[1])
                    with lock:
                    # 向所有其他客户端广播数据
                        for client in clients:
                            if client == client_socket:
                                my_dict = {"register": v}  # create your dictionary here
                                json_data = json.dumps(my_dict)  # convert dictionary to json
                                json_data +='\n'
                                client.sendall(json_data.encode('utf-8'))  # send json data
                    #queue.put(v)
                    if v!='error':
                        send_message_to_db(f"ip:{client_address[0]}, user :{v} register success.")
                        fdb.put('/user_data/'+u,"nowlogin",0)
                        fdb.put('/','ppt',7)
                    else:
                        send_message_to_db(f"ip:{client_address[0]}, user :{v} register fail.")
                        fdb.put('/','ppt',6)
                        #time.sleep(1)
                        fdb.put('/','ppt',8)
                elif (command[0]=="login"):
                    fdb.put('/','ppt',5)
                    v = login()
                    with lock:
                    # 向所有其他客户端广播数据
                        for client in clients:
                            if client == client_socket:
                                my_dict = {"login": v}  # create your dictionary here
                                json_data = json.dumps(my_dict)  # convert dictionary to json
                                json_data +='\n'
                                client.sendall(json_data.encode('utf-8'))  # send json data
                    #queue.put(v)
                    u = v
                    print(f'ip:{client_address[0]}, user:{u} has login')
                    if u!='error':
                        send_message_to_db(f'ip:{client_address[0]}, user:{u} has login.')
                        fdb.put('/user_data/'+u,"nowlogin",1)
                        fdb.put('/user_data/'+u,"ip",client_address[0])
                        fdb.put('/','ppt',7)
                        time.sleep(1)
                        fdb.put('/','ppt',8)
                    else:
                        send_message_to_db(f'ip:{client_address[0]}, fail to login.')
                        fdb.put('/','ppt',6)
                        time.sleep(1)
                        fdb.put('/','ppt',8)
                elif (command[0]=="face_indentify"):
                    fdb.put('/','ppt',25)
                    v = identify(command[1])
                    v['消費紀錄']=""
                    fdb.put('/user_data/'+command[1],'score',v)
                    buy = fdb.get('/user_data/'+command[1]+'/buy',None)
                    for times,item in buy.items():
                        for name,price in item.items():
                            v[times]="買"+name+" "+str(price)
                    with lock:
                    # 向所有其他客户端广播数据
                        for client in clients:
                            if client == client_socket:
                                my_dict = v  # create your dictionary here
                                json_data = json.dumps(my_dict)  # convert dictionary to json
                                json_data +='\n'
                                client.sendall(json_data.encode('utf-8'))  # send json data
                    time.sleep(5)
                    fdb.put('/','ppt',8)
                elif(command[0]=='speak'):
                    fdb.put('/','ppt',9)
                    v=reply(command[1],command[2])
                    if v is None:
                        v=openai_api(command[2])
                    print("speak mes",v)
                    with lock:
                    # 向所有其他客户端广播数据
                        for client in clients:
                            if client == client_socket:
                                my_dict = {"speak": v}  # create your dictionary here
                                json_data = json.dumps(my_dict)  # convert dictionary to json
                                json_data +='\n'
                                client.sendall(json_data.encode('utf-8'))  # send json data
                    time.sleep(3)
                    fdb.put('/','ppt',8)
                elif(command[0]=='menu'):
                    vs=command[2].split(";")
                    for v in vs:
                        firebase_.insert_wine(command[1],v)
                        fdb.put('/','ppt',dic[v])
                        time.sleep(1)
                    time.sleep(2)
                    fdb.put('/','ppt',8)
                elif(command[0]=='order'):
                    firebase_.insert_wine(command[1],command[2])
                    fdb.put('/','ppt',dic[command[2]])
                    time.sleep(3)
                    fdb.put('/','ppt',8)
                elif(command[0]=='ppt'):
                    fdb.put('/','ppt',int(command[1]))
            else:
                # 客户端断开连接
                remove_client(client_socket)
                print(f"Client {client_address} disconnected")
                send_message_to_db(f"Client {client_address} disconnected")
                break
        except ConnectionResetError:
            # 客户端强制断开连接
            remove_client(client_socket)
            print(f"Client {client_address} forcibly disconnected")
            break

def broadcast_data(data, sender_socket):
    # 向除了发送者之外的所有客户端广播数据
    with lock:
        print(111)
        for client in clients:
            if client != sender_socket:
                message = data.decode('utf-8').strip()
                message +='\n'
                print(client,data)
                client.sendall(message.encode('utf-8'))

def remove_client(client_socket):
    # 从列表中移除断开连接的客户端
    with lock:
        if client_socket in clients:
            clients.remove(client_socket)

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f"Server started on {HOST}:{PORT}")
    url = "https://python-database-3b3f8-default-rtdb.firebaseio.com/"
    fdb = firebase.FirebaseApplication(url, None)
    fdb.put('/','ppt',1)
    while True:
        client_socket, client_address = server_socket.accept()
        with lock:
            clients.append(client_socket)
        print(f"New client connected: {client_address}")
        url = "https://python-database-3b3f8-default-rtdb.firebaseio.com/"
        fdb = firebase.FirebaseApplication(url, None)
        tmp = fdb.get('/user/tmpip',None)
        if tmp!=client_address[0]:
            fdb.put('/user','nowip',client_address[0])
            fdb.put('/user','tmpip',client_address[0])
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_thread.start()

if __name__ == '__main__':
    main()
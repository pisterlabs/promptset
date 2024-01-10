import openai
import os
import socket, threading
import struct
from typing import Type
import asyncio

class Server :
    def __init__(self,gpt, whisper) :
        self.client_thread = list()
        self.Whisper = whisper
        self.GPT = gpt
    
    def ConnectClient(self) :
        print("Try Connecting")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('',23162))
        self.server_socket.listen()

        while True :
            self.client_socket, self.addr = self.server_socket.accept()
            if self.client_socket is not None : 
                 print("연결됨")
                 break
        asyncio.run(self.HandleClient())
    
    async def HandleClient(self) :
        print("Connecting Success")
        print(self.client_socket)
        while True :
            data = self.Recv()
            id = int.from_bytes(data,'big')
            if id == 5 :
                text = ""
                npcName, waves = self.RecvVoicePacket()
                print(npcName+"잘받았습니다.")
                for wave in waves :
                    text += self.Whisper(wave)
                print(text)
                response, action = self.GPT(npcName, text) 
                self.SendGPTPacket(response,action)
            if id == 2 : 
                msg = await self.RecvMessagePacket()
                print(msg)



    def RecvVoicePacket(self) :
        try : 
            waves = []
            byte = self.Recv()
            npcName = byte.decode('utf-8')
            byte = self.Recv()
            arraycount = int.from_bytes(byte,"big")
            for i in range(0,arraycount) :
                bytedata = self.Recv()
                data = []
                for i in range(0,len(bytedata)-4,4) :
                    value = struct.unpack('>f', bytedata[i:i+4])[0]
                    data.append(value)
                waves.append(data)
        except Exception as e: 
            print(f"no voice packet{0}",e)
        return npcName, waves
    
    async def RecvMessagePacket(self) :
        try :
            byte = self.Recv()
            data = byte.decode('utf-8')
            return data
        except Exception as e :
            print(f"no Message Packet {0}",e)

    def RecvChatPacket(self) :
        who = self.RecvMessagePacket()
        self.RecvVoicePacket()
        
    def SendGPTPacket(self,response, action) :
        self.Sendint(1)
        self.Send(response)
        self.Send(action)
    
    def Sendint(self,id) :
        bytes = struct.pack('>i',id)
        self.client_socket.sendall(struct.pack('>i',len(bytes)))
        self.client_socket.sendall(bytes)

    def Send(self,msg):
        bytes = msg.encode()
        self.client_socket.sendall(struct.pack('>i',len(bytes)))
        self.client_socket.sendall(bytes)
   
    def Recv(self) -> bytes :
        try:
            byte = self.client_socket.recv(4)
            len = int.from_bytes(byte,"big")
            byte = self.client_socket.recv(len)
            return byte
        except :
            print("no receive")



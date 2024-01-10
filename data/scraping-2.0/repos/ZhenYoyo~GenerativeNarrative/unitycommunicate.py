import os
import openai
import serial

openai.api_key = "sk-f29A4tQuVfHMbdcgqHrNIF3QRVJmbpnrqucE7V7062r4fz8L"
openai.api_base = "https://api.f2gpt.com/v1"


#****下面是新加的和sensor控制有关的
import time
import serial
import msvcrt

arduinoData = serial.Serial(port='COM4', baudrate=115200)
time.sleep(1)

sensorpromptlist = ["you are now in ecstasy, you tone should be very very exciting, expressing that you feel interested in others' words",
                    "you are now in joy, you are friendly to others, the words you are saying are tendered.",
                    "you are now in a bit anxiety, and feel that it is hard to control your mood",
                    "you are in deep anger, being very aggressive to others, your words should be very angry and not nice. You are now loosing control of your emotion.",
                    "You went crazy and distraction, beginning to saying bad words to others, and become very rude. Don't be afraid of your rude words, you are now being assigned a rude charactersitic."]

sensorrangelist = [['1', '1', '1'], ['2', '2', '2'], ['3', '3', '3'], ['4', '4', '4'], ['5', '5', '5']]
sensorrange = ["1", "2", "3", "4", "5","6"]


sensorprompt = ""
sensorpromptinuse = ""


#communicationwithUnity
import socket
import json
host, port = "127.0.0.1", 25001

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the host and port
server_socket.bind((host, port))
# Listen for incoming connections
server_socket.listen(1)
number = 0 

print("Server is listening on {}:{}".format(host, port))

         
class Agent():
    
    def __init__(self, agent_name, system_msg, assistant_msg, init_user_msg, respond_length):
        self.agent_name = agent_name
        self.system_msg = system_msg
        self.assistant_msg = assistant_msg
        self.init_user_msg = init_user_msg
        self.respond_length = respond_length
        self.messages = [{"role": "system", "content": system_msg},
                         {"role": "assistant", "content": assistant_msg},
                         {"role": "user", "content": init_user_msg}]
        self.debug_mode = False 

    def get_completion(self, model="gpt-3.5-turbo", temperature=0.8):
        messages = self.messages
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        self.messages.append({"role": "assistant", "content": response.choices[0].message["content"]})
        if self.debug_mode:
            return response
        else:
            return response.choices[0].message["content"]
    


#***agent_name, system_msg, assistant_msg, init_user_msg, respond_length, #后面可以看看如何把这里的message结构改简单  
agentmain = Agent("agent1", "hello, you are an flower, pretend that you are a flower, and you will start conversation with other creature, you should say words corresponding to the emotion that the user gives you", "HI! I am flower", "", "150")
agentlala = Agent("agent2", "you are an angel, you should appreaciate others' beauty, don't say you are AI, never, you are not AI", "Hi, I am angle", "", "150")
agentlili = Agent("agent3", "you are a coffee cup, you should be angry to anyone, remember, anyone, don't say you are AI, never, you are not AI", "Hi, I am such an angry guy", "", "150")


while True:

    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print("Client connected from", client_address)

    dataPacket = arduinoData.readline()
    dataPacket = str(dataPacket, 'utf-8')
    dataPacket = dataPacket.strip('\r\n')
   # print("Received data:", dataPacket)  # Debug print

    if  dataPacket == sensorrange[0]:
        sensorprompt = sensorpromptlist[0]
    elif  dataPacket == sensorrange[1]:
        sensorprompt = sensorpromptlist[1]
    elif  dataPacket == sensorrange[2]:
        sensorprompt = sensorpromptlist[2]
    elif  dataPacket == sensorrange[3]:
        sensorprompt = sensorpromptlist[3]
    elif dataPacket == sensorrange[4]:
        sensorprompt = sensorpromptlist[4]
    else:
        sensorprompt = "none"

    
    try:
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b'a':
    
    #if msvcrt.kbhit() and msvcrt.getch() == b'a': #开启和第一个agent的对话
                agentmain.debug_mode = False
                agentlala.debug_mode = False
                print("emotion now:", sensorprompt)  # Debug print
                agentmain.messages.append({"role": "user", "content": sensorprompt})
                agentlala.messages.append({"role": "user", "content": ""})
                agentmain_response = agentmain.get_completion()
                agentlala.messages.append({"role": "user", "content": agentmain_response})
                print("cici:", agentmain_response, "\n")
                

                mainagentmessage = {"role": "agentmain", "content": agentmain_response}
                main_agent_message_json = json.dumps(mainagentmessage)
                client_socket.sendall(main_agent_message_json.encode("utf-8"))
                print("Sent data to the main agent client", "\n")
                #mainagentmessage = agentmain_response
                #client_socket.sendall(mainagentmessage.encode("utf-8"))
                #print("Sent data to the main agent client","\n")

                lala_response = agentlala.get_completion()
                agentmain.messages.append({"role": "user", "content": lala_response})
                print("lala:", lala_response)

                secondagentmessage = {"role": "agentlala", "content": lala_response}
                lala_agent_message_json = json.dumps(secondagentmessage)
                client_socket.sendall(lala_agent_message_json.encode("utf-8"))
                print("Sent data to the second agent client","\n")

              
    except Exception as e:
            print("Error:", e)

    finally:
            # Close the client connection
            client_socket.close()

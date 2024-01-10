##################################################################################################
# About: Server TCP code, get json object from the simulator and send another json object and
#       process the data using stable baseline
# Notes:
#TODO (DONE): Adapt this on python3 to solve the issues in json parser in python3
#there is difference in json.loads as it only accept string not bytes  and in 3 TCP read return bytes
#and str not converting from bytes to str in py3 but .decode('UTF-8') does
#and the same for sendall function of TCP it only takes bytes so we need to encode the string first to bytes like object
#and solving some errors like https://bugs.python.org/issue24283

#Reference: https://pymotw.com/2/socket/tcp.html

#Coding Style: camelCase
# Run it with . ~/virtualenvs/baselines_env/bin/activate

##################################################################################################
#import the libraries
import socket
import sys
import signal
import json
from time import *
import os
import random
import numpy as np
from transforms3d.euler import euler2mat

# import stable_baselines


print("Finish importing the libraries")

#import openai
#import tensorflow as tf
#import numpy as np
#from baselines import ...

#--------------------------------------------Vars--------------------------------------------

#Settings for the TCP communication
packetSize = 500
portNum = 10008
hostName = 'localhost'
# connection = None
# clientAddress = None

globalFlag = 0  #this is used to reset the NTRT environment and TCP connection with it

# JSON object structure
jsonObj = {
    # 'Controllers_num': 9,
    # 'Controllers_index': [2, 4, 5, 6, 7, 11, 13, 17, 19],
    # 'Controllers_val': [18,-1,-1,-1,-1,-1,-1,-1,-1],
    # 'Controllers_val': [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    'Controllers_val': [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
    'Reset': 0
}
#--------------------------------------------------------------------------------------------


#--------------------------------------------Functions--------------------------------------------
# Ctrl+C Handle to close safely the TCP connection
def signalHandler(signal, frame):
    # print('You pressed Ctrl+C!')
    tmp = str(input("You want reset or close: r/c: \n"))
    print(tmp)
    if(tmp == 'r'):
        reset()
    elif(tmp == 'c'):
        print("----------------------------------Exit-----------------------------------")
        global globalFlag
        globalFlag = 2
    else:
        # print("Please  Ctrl+C and write 'r' or 'c' ")
        sleep(5)
# function for writing data into TCP connection
def write(connection, data):
    # print('sending data to the client:"{}"'.format(data))
    try:
        connection.sendall(data.encode())
    except Exception as e:
        print("$$$$$$$$$$$$ ERROR in Writing $$$$$$$$$$$$")
        print("Error: " + str(e))

# function for reading data from TCP connection
def read(connection):
    try:
        data = []
        counter = 1
        # Receive the data in small chunks and retransmit it
        while True:
            data.append(connection.recv(packetSize).decode("utf-8"))         #reading part
            # print('{} received "{}"'.format(counter,data[-1]))
            # print(data[-1][-14:-1], ('ZFinished' in str(data[-1][-14:-1])))
            if 'ZFinished' in str(data[-1][-14:-1]):
                # print("FINISHED*************")
                # sleep(5)
                break
            counter += 1
        return "".join(data)
    except ValueError:
        print(ValueError)
        print("$$$$$$$$$$$$ ERROR in Reading $$$$$$$$$$$$")
        # sleep(2)
        return None
def reset():
    global globalFlag
    globalFlag = 1

def main():
    start_time = time()
    while True:
        #Note: TODO: Make in the simulator wait a second then send a message
        os.system('/home/hany/repos/Work/IU/Tensegrity/Tensegrity-Robotics/src/dev/legz/python_communication_test/helper.sh')

        print('#########\nwaiting for a connection\n#########')
        connection, clientAddress = sock.accept()  #wait until it get a client
        print('connection from', clientAddress)
        global globalFlag
        globalFlag = 0
        target = 24
        sign = -5

        while True:
            r = read(connection)
            # print(r)
            if(r != None):
                jsonObjTmp = json.loads(r)  # Parse the data from string to json
            print("s1##{:} $${:}".format(jsonObj["Controllers_val"][2],jsonObjTmp["Controllers"][2]))

            # TODO: Use the incoming data after being converted to json

            # TODO:
            # Take the data from the simulator module
            # Formulate the data as observation
            # Generate Reward
            # Feed the RL Algorithm with Reward and observartion
            # Generate Action
            # Decide either end of episode (Reset the simulator) or specific Action
            # Modify the action in json
            


            # if(jsonObjTmp["Controllers"][2] >= 23.5 and sign == 1):
            #     print("FLIP")
            #     target = jsonObjTmp["Controllers"][2]
            #     sign = -6
            # if(jsonObjTmp["Controllers"][2] <= 22.5 and sign == -6):
            #     print("FLIP")
            #     # target = 24
            #     sign = 1
            #     target = jsonObjTmp["Controllers"][2] + sign*0.5
            # # print(target)
            # print(sign)
            # # jsonObj["Controllers_val"][2] = target
            # if(jsonObjTmp["Flags"][0] == 1):
            #     print("FLAG")
            #     # jsonObj["Controllers_val"][2] = target
            #     jsonObj["Controllers_val"][2] = jsonObjTmp["Controllers"][2]

            # print("s2##{:} $${:}".format(jsonObj["Controllers_val"][2],jsonObjTmp["Controllers"][2]))
            # input()
            # # jsonObj["Controllers_val"][2] = jsonObjTmp["Controllers"][2]
            # if((time() - start_time)% 5 and jsonObjTmp["Flags"][0] == 1):
            # print(jsonObjTmp["Center_of_Mass"][4], jsonObjTmp["Orientation"][4])
            # CMS = np.array(jsonObjTmp["Center_of_Mass"][4])
            # half_length = 15
            # orientation_vector = np.array(jsonObjTmp["Orientation"][4][:3])
            # end_point_local1 = np.array([0, half_length,0])
            # end_point_local2 = np.array([0,-half_length,0])

            # yaw,pitch,roll = orientation_vector
            # rot_mat = np.matrix(euler2mat(yaw, pitch, roll, 'syxz'))
            # print(rot_mat)

            # # print("end_point1 in local coordinate system", end_point_local1)
            # # print("end_point2 in local coordinate system", end_point_local2)


            # end_point_world1 = CMS+rot_mat.transpose().dot(end_point_local1)
            # end_point_world2 = CMS+rot_mat.transpose().dot(end_point_local2)


            # print("#2 end_point1 in world coordinate system", end_point_world1)
            # print("#2 end_point2 in world coordinate system", end_point_world2)


            if(jsonObjTmp["Flags"][0] == 1):
                sign = -1*sign
                print("FLIP")

            jsonObj["Controllers_val"][2] = sign
            jsonObj["Controllers_val"][5] = sign

            print("state##{:} $${:}".format(jsonObj["Controllers_val"][2],jsonObjTmp["Controllers"][2]))
            
                
            write(connection,json.dumps(jsonObj))   # Write to the simulator module the json object with the required info
            if(globalFlag > 0):
                print("GLOBAL FLAG Exit")
                break
        connection.close()
        if(globalFlag == 2):
            sys.exit(0)
#-------------------------------------------------------------------------------------------------


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # Create a TCP/IP socket
serverAddress = (hostName, portNum) # Bind the socket to the port


print('#########\nstarting up on {} port {}\n#########'.format(serverAddress, portNum))
sock.bind(serverAddress)
sock.listen(1)  # Listen for incoming connections



signal.signal(signal.SIGINT, signalHandler) # Activate the listen to the Ctrl+C


# This is top open the simulator
print("Opening the NTRT simulator")

main()
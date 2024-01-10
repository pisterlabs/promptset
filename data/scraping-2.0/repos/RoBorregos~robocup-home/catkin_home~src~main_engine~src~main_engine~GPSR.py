#!/usr/bin/env python3
import rospy
import actionlib
import time
from enum import Enum
from std_msgs.msg import String
from std_msgs.msg import Bool, Int32
from std_srvs.srv import SetBool
from sensor_msgs.msg import LaserScan
# from object_detector.msg import DetectObjects3DAction, DetectObjects3DGoal, objectDetectionArray, objectDetection
# from pick_and_place.msg import PickAndPlaceAction, PickAndPlaceGoal
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, PointStamped, Point, Quaternion
from nav_msgs.msg import Odometry
import tf
import tf2_ros
from sensor_msgs.msg import JointState
from tf2_geometry_msgs import do_transform_pose
from tf import transformations
from tf.transformations import quaternion_from_euler
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from detection_msgs.msg import BoundingBoxes, BoundingBox
from action_selectors.msg import RawInput
from humanAnalyzer.msg import face_array


import os
import openai
import json
import numpy as np

ARGS= {
    "VERBOSE": True,
}

OBJECTS_NAME= {
    1 : 'COKE',
    2 : 'APPLE',
    3 : 'MUG',
    4 : 'SOAP',
    5 : 'BANANA',
    6 : 'JAR'
}

OBJECTS_ID= {
    'COKE' : 1,
    'APPLE' : 2,
    'MUG' : 3,
    'SOAP' : 4,
    'BANANA' : 5,
    'JAR' : 6
}

promts = {
    "start": "Let's start General Purpose Service Robot ",
    "wait": "I am waiting for someone",
    "self_intro": "Hello, I'm homie, your personal assistant.",
    "ask_comm": "What do you want me to do ?",
    "ask_drink": "What is your favorite drink?",
    "repeat": "i think i didn't get that, can you repeat it?",
    "come" : "come with me ",
    "unreached": "I am sorry, I couldn't understand your command, pleasy try again ",
    "sorry" : "I am sorry, I think i didn't get that",
    "assign": "Can you please sit on the chair in front of me?"
}

calls = {
    "get_name": "if there is nos name return ''get me the name of the person on the next message: ",
    "get_drink": "get me the name of the beverage on the next message: ",
    "describe": "describe a person with the next attributes: ",
    "confirm" : "tell me onlyTrue or False, the next message is a general confirmation, like ues, OK, got it:",
    "reached": "tell me only True or False, the next message is a confirmation that you reached a place:",

    "get_loc": "the valid locations are living room, dining room, kitchen, bedroom. Give me only the location on this message: ",
    "get_obj": "the valid objects are coke, apple, mug, soap, banana, pitcher. Now give me only the object on this message: ",
    "get_per": " the valid names are Jamie, Morgan, Micheal, Jordam, Taylor, Tracy, Robin, Alex. Now give me only the name on this mesage: "
}



GoTo = 1
Find = 2
FindPerson = 3
GotoBring = 4
openai.api_key = "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"

json_path = "/home/kevin/Desktop/workspace/home/TMR2023/robocup-home/catkin_home/src/humanAnalyzer/src/scripts/identities.json"

INITIALIZATION_STATE = "init"
START_STATE = "start"
WAIT_STATE = "waiting for someone"
COMMANDER = "Recieeving commands"
CONIRMATION = "confirm the command"
DOING = "Do the command"
END_STATE = "done"


class DoorDetector:
    def __init__(self):
        self.last_distance = None
        self.door_opened = False

        self.firstIteration = True
        # Subscribe to the LiDAR data topic
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
                       # '/scan'
        # Create a publisher for the movement command topic
        self.cmd_pub = rospy.Publisher('/move_command', String, queue_size=10)

    def lidar_callback(self, msg):
        # Filter the LiDAR data to only include points near the door
        door_angle = 1.6  # Replace with the angle of the door in the LiDAR data (rad) 1.6 is straight of our robot
        #door_distance = 0.3  # Replace with the distance of the door in the LiDAR data (m)
        door_idx = int(door_angle / msg.angle_increment)
        min_idx = max(0, door_idx - 10)
        max_idx = min(len(msg.ranges), door_idx + 10)
        door_ranges = msg.ranges[min_idx:max_idx]

        # Calculate the median distance to the door
        door_distance = sorted(door_ranges)[len(door_ranges)//2]
        #rospy.loginfo("Door: %f", door_distance)
        # Check if the distance has changed significantly since the last scan
        if self.last_distance is not None and abs(door_distance - self.last_distance) > 0.30:   #> #.## (m) how much difference from door_distance
            self.door_opened = True
        self.last_distance = door_distance

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.door_opened:
                return True
            rate.sleep()

        

class gpsr(object):
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.currentState =  INITIALIZATION_STATE
        self.iteration = 0

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move base server")
        f = open('/home/kevin/Desktop/workspace/home/TMR2023/robocup-home/catkin_home/src/main_engine/src/main_engine/nav.json')
        self.map_context = json.load(f)
        # print(json.dumps(self.map_context))
        f.close()

        # 
        self.detectionSub = rospy.Subscriber('detections/output', BoundingBoxes, self.detectionCallback)
        self.detectedObjects = []
        
        self.detectionPersonSub = rospy.Subscriber('idPerson', Int32, self.detectionPersonCallback)
        self.detectedPerson = None
        
        # GPT API
        self.GPT_model="text-davinci-003"
        self.GPT_temperature=0.7
        self.GPT_top_p=1
        self.GPT_frequency_penalty=0
        self.GPT_presence_penalty=0
        rospy.logdebug("GPT API initialized")
        #Conversation initialization
        self.inputText = ""
        self.speech_enable = rospy.Publisher("inputAudioActive", Bool, queue_size=10)
        self.saying = False
        self.say_listener = rospy.Subscriber("saying", Bool, self.say_callback)
        self.speech_enable.publish(Bool(False))
        self.say_publisher = rospy.Publisher('robot_text', String, queue_size=20)
        self.input_suscriber = rospy.Subscriber("RawInput", RawInput, self.input_callback)

        #Nav initialization 
        #self.cancel_move = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        #self.initial_pose = None
        #self.initial_pose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose_cb)
        self.is_moving = False

        #Face detection

        self.faces_subscriber = rospy.Subscriber("/faces", face_array, self.face_callback)
        rospy.loginfo("Receptionist initialized")

        # Conversation dependencies
        self.detectedObjects = []
       


        self.currentState = START_STATE
        # wait for the robot to be ready
        time.sleep(1)
        # start the main loop
        # 
        # Array of persons
        self.faces = None
        self.persons = []

        # initialize embeddings and command variables

        self.embeddings = np.load("/home/kevin/Desktop/workspace/home/TMR2023/robocup-home/catkin_home/src/main_engine/src/main_engine/GoTo_embedding.npz")
        self.GoTo_embedding = self.embeddings['GoTo']
        self.Find_embedding = self.embeddings['Find']
        self.FindPerson_embedding = self.embeddings['FindPerson']
        self.GotoBring_embedding = self.embeddings['GotoBring']

        self.cm_location = None
        self.cm_obj = None
        self.cm_person = None
        self.cm_pick = False

        self.run()
        rospy.spin()

    def detectionPersonCallback(self, msg):
        # 1 Morgan - Emiliano
        # 2 Michael - Ivan
        if  msg.data == 1:
            self.detectedPerson = "Morgan"
        elif msg.data == 2:
            self.detectedPerson = "Michael"
        else:
            self.detectedPerson = None
    
    def detectionCallback(self, boundingBoxes_):
        for box in boundingBoxes_.bounding_boxes:
            if box.probability > 0.85 and box.Class not in self.detectedObjects:
                self.detectedObjects.append(box.Class.lower())
        rospy.logwarn("Detected objects: " + str(self.detectedObjects))

    def execute_command(self, type_, location_, object_, person_):
      rospy.logwarn("Executing command")
      if type_ == 4:
          type_ = 2
      if type_ == 1: # Go to Place
        for key in self.map_context:
          rospy.logwarn("Key in for: " + str(key))
          rospy.logwarn("locatio: " + str(location_))
          if key.lower() == location_.lower():
            rospy.logwarn("Key inside if" + str(key))
            pose = MoveBaseGoal()
            pose.target_pose.header.frame_id = 'map'
            pose.target_pose.pose.position.x = self.map_context[key]["safe_place"][0]
            pose.target_pose.pose.position.y = self.map_context[key]["safe_place"][1]
            pose.target_pose.pose.position.z = self.map_context[key]["safe_place"][2]
            pose.target_pose.pose.orientation.x = self.map_context[key]["safe_place"][3]
            pose.target_pose.pose.orientation.y = self.map_context[key]["safe_place"][4]
            pose.target_pose.pose.orientation.z = self.map_context[key]["safe_place"][5]
            pose.target_pose.pose.orientation.w = self.map_context[key]["safe_place"][6]
            self.move_base_client.send_goal(pose)
            res = self.move_base_client.wait_for_result()
            break
      if type_ == 2: # Go to Place and find object
        for key in self.map_context:
          if key.lower() == location_.lower():
            for place in self.map_context[key]:
                pose = MoveBaseGoal()
                pose.target_pose.header.frame_id = 'map'
                pose.target_pose.pose.position.x = self.map_context[key][place][0]
                pose.target_pose.pose.position.y = self.map_context[key][place][1]
                pose.target_pose.pose.position.z = self.map_context[key][place][2]
                pose.target_pose.pose.orientation.x = self.map_context[key][place][3]
                pose.target_pose.pose.orientation.y = self.map_context[key][place][4]
                pose.target_pose.pose.orientation.z = self.map_context[key][place][5]
                pose.target_pose.pose.orientation.w = self.map_context[key][place][6]
                # self.moveToPose(pose)
                self.move_base_client.send_goal(pose)
                res = self.move_base_client.wait_for_result()

            rospy.sleep(5)
            import time
            x = time.time()
            success = False
            self.detectedObjects = []
            rate = rospy.Rate(10)
            while (time.time() - x < 5.0):
                if object_.lower() in self.detectedObjects:
                    success = True
                    self.say("Object Found")
                    break
                rospy.logwarn(object_.lower() + " not found")
                rate.sleep()

            if not success:
                pass
                # self.say("Not here, i'll keep looking")

      if type_ == 3: # Go to Place and find person
        for key in self.map_context:
          if key.lower() == location_.lower():
            for place in self.map_context[key]:
                pose = MoveBaseGoal()
                pose.target_pose.header.frame_id = 'map'
                pose.target_pose.pose.position.x = self.map_context[key][place][0]
                pose.target_pose.pose.position.y = self.map_context[key][place][1]
                pose.target_pose.pose.position.z = self.map_context[key][place][2]
                pose.target_pose.pose.orientation.x = self.map_context[key][place][3]
                pose.target_pose.pose.orientation.y = self.map_context[key][place][4]
                pose.target_pose.pose.orientation.z = self.map_context[key][place][5]
                pose.target_pose.pose.orientation.w = self.map_context[key][place][6]
                self.moveToPose(pose)
                self.move_base_client.send_goal(pose)
                self.move_base_client.wait_for_result()
            rospy.sleep(5)
            import time
            x = time.time()
            success = False
            while (time.time() - x < 5.0):
                if person_ == self.detectedPerson:
                    success = True
                    self.say("Person Found")
                    break
            if not success:
                self.say("Not here, i'll keep looking")

    def moveToPose(self, pose):
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = 'map'
        move_base_goal.target_pose.pose = pose
        rospy.logwarn("Before movebase")
        self.move_base_client.send_goal(move_base_goal)
        rospy.logwarn("Sended")
        res = self.move_base_client.wait_for_result()
        rospy.logwarn(res)
    
    def moveTo(self, option):
        move_base_goal = MoveBaseGoal()
        if option == 1: 
            move_base_goal.target_pose.header.frame_id = 'map'
            move_base_goal.target_pose.pose.position.x = 2.59
            move_base_goal.target_pose.pose.position.y = -0.1526
            move_base_goal.target_pose.pose.position.z = 0.0
            move_base_goal.target_pose.pose.orientation.x = 0
            move_base_goal.target_pose.pose.orientation.y = 0
            move_base_goal.target_pose.pose.orientation.z = 0.99
            move_base_goal.target_pose.pose.orientation.w = 0.0678
        else: 
            move_base_goal.target_pose.header.frame_id = 'map'
            move_base_goal.target_pose.pose.position.x = 2.805
            move_base_goal.target_pose.pose.position.y = -0.3707
            move_base_goal.target_pose.pose.position.z = 0.0
            move_base_goal.target_pose.pose.orientation.x = 0
            move_base_goal.target_pose.pose.orientation.y = 0
            move_base_goal.target_pose.pose.orientation.z = -0.782
            move_base_goal.target_pose.pose.orientation.w = 0.622
        self.move_base_client.send_goal(move_base_goal)
        self.move_base_client.wait_for_result()

    def get_embedding(self, text, model="text-embedding-ada-002"):
        rospy.logwarn(text)
        text = text.replace("\n", " ")
        rospy.logdebug(text)
        return openai.Embedding.create(input = [str(text)], model=model)['data'][0]['embedding']
                
    def say_callback(self, data):
        self.saying = data.data
        rospy.logwarn("saying: " + str(self.saying))

    def face_callback(self, data):
        self.faces = data.faces
    
    def confirmation(self):
        return True

    def goToOrigin(self):
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = 'map'
        move_base_goal.target_pose.pose.position.x = 2.46
        move_base_goal.target_pose.pose.position.y = -0.33
        move_base_goal.target_pose.pose.position.z = -0.5
        move_base_goal.target_pose.pose.orientation.x = 0
        move_base_goal.target_pose.pose.orientation.y = 0
        move_base_goal.target_pose.pose.orientation.z = 0
        move_base_goal.target_pose.pose.orientation.w = 1
        rospy.loginfo('Sending move_base goal: {}'.format(move_base_goal))
        self.move_base_client.send_goal(move_base_goal)
        self.move_base_client.wait_for_result()

    def run(self):
        self.goToOrigin()
        while not rospy.is_shutdown():
            rospy.loginfo("-----------------------------------------RUN-----------------------------------------")

            if self.currentState == START_STATE:
                self.speech_enable.publish(Bool(False))

                rospy.loginfo("GPSR is ready to start")
                self.say(promts["start"])               
                self.currentState = WAIT_STATE

                
            elif self.currentState == WAIT_STATE:

                if self.is_someone():
                    self.say(promts["self_intro"])
                    self.currentState = COMMANDER

                else:
                    self.say("I cant see you")
        

                
            elif self.currentState == COMMANDER:
                command = self.get_command()
                

                if command != "":
                   
                    # if  name not in self.persons.name:
                            
                    self.currentState = CONIRMATION
                else:
                    self.currentState = COMMANDER

            elif self.currentState == CONIRMATION:
                rospy.logwarn("INSIDE CONFIRMATION")
                if self.confirmation():
                    self.currentState = DOING
                else:
                    self.currentState = COMMANDER

                    self.say(promts["unreached"])
        
            elif self.currentState == DOING:
                rospy.logwarn(" The command is " + str(command) + " " + str(self.cm_location) + str(self.cm_obj) + str(self.cm_person)  )
                self.say(" The command is " + str(command) + " " + str(self.cm_location) + str(self.cm_obj) + str(self.cm_person))
                self.execute_command(command,self.cm_location,self.cm_obj, self.cm_person)
                self.goToOrigin()
                self.currentState = COMMANDER

            elif self.currentState == END_STATE:
                self.say(promts["end"])        


    def say(self, text):
        time.sleep(1)   
        if self.saying == False:
            self.saying == True
            self.say_publisher.publish(text)
        time.sleep(len(text)/8 )
    


    def get_command(self):
        self.speech_enable.publish(Bool(False))

        command = ""
        self.inputText = ""
        # get the command from the user
        rospy.loginfo("Receptionist is getting command")
        i = 0
        # wait for the user to say something
        rospy.logwarn("Waiting for user input")
        self.say(promts["ask_comm"])
        self.speech_enable.publish(Bool(True))
        rospy.logwarn("command : " + command)
        rospy.logwarn(command  != "" and not rospy.is_shutdown())
        while not rospy.is_shutdown() and command == "":
            command = self.inputText
            i+=1
            time.sleep(1)
            rospy.logdebug(command)
            if self.inputText != "":
                    command_embedding = self.get_embedding(self.inputText[:-1])

                    cosine_similarity = lambda x, y: np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

                    nearest = 0
                    nearest_command = ""
                    for emb, command in zip([self.GoTo_embedding, self.Find_embedding, self.FindPerson_embedding, self.GotoBring_embedding], [GoTo, Find, FindPerson, GotoBring]):
                        similarity = cosine_similarity(emb, command_embedding)
                        rospy.logwarn("------------------------------------")
                        rospy.logwarn(str(similarity) + " " + str(command))
                        if similarity > nearest:
                            nearest = similarity
                            nearest_command = command

                    rospy.loginfo("nearest: " +str(nearest_command))

                    if nearest_command == 1:
                        self.cm_person = None
                        self.cm_obj = None
                        self.cm_location = self.parser(calls["get_loc"])
                        self.cm_pick = False

                        
                    elif nearest_command == 2: 
                        self.cm_location = self.parser(calls["get_loc"])
                        self.cm_obj = self.parser(calls["get_obj"])
                        self.cm_person = None
                        self.cm_pick = False

                    elif nearest_command == 3:
                        self.cm_location = self.parser(calls["get_loc"])
                        self.cm_person = self.parser(calls["get_name"])
                        self.cm_obj = None
                        self.cm_pick = False

                    
                    elif nearest_command == 4:
                        self.cm_location = self.parser(calls["get_loc"])
                        self.cm_obj = self.parser(calls["get_obj"])
                        self.cm_person = None
                        self.cm_pick = True
                    


                    if self.cm_location == "living room":
                        self.cm_location = "living_room"
                    if self.cm_location == "dinner room":
                        self.cm_location = "dinner_room"
                    self.cm_location = self.cm_location.strip() if self.cm_location is not None else None
                    self.cm_obj = self.cm_obj.strip() if self.cm_obj is not None else None
                    self.cm_person = self.cm_person.strip() if self.cm_person is not None else None

            if i%28 == 0:
                self.speech_enable.publish(Bool(False))
                self.say(promts["ask_comm"])
                self.speech_enable.publish(Bool(True))
                i = 0
                
        rospy.logwarn("command2: " + str(command))
        rospy.logwarn(self.cm_location)
        rospy.logwarn(self.cm_person)
        rospy.logwarn(self.cm_obj)



        self.speech_enable.publish(Bool(False))

        rospy.logwarn(command)

        time.sleep(3)
        return nearest_command
    

    def go_to(self, place):
        self.is_moving = True
        rospy.loginfo("Receptionist is going to " + place)
        time.sleep(10)
        return True

    def get_attributes(self, which = 1):

        with open(json_path, 'r') as f:
            data = json.load(f)
            f.close()

        # print(data[list(data.keys())[0]])
        # print(list(data.keys())[0])
        print("size: " + str(list(data.keys()).__len__()))
        size = list(data.keys()).__len__()
        my_p = person()
        my_p.id = list(data.keys())[size-which]
        # print("drink: "+ list(data.keys())[0])
        # print("drink: "+ str(data[my_p.id]["age"]))
        try:
            my_p.age = data[my_p.id]["age"]
        except:
            pass
        try:
            my_p.race = data[my_p.id]["race"]
        except:
            pass
        try:
            my_p.gender = data[my_p.id]["gender"]
        except:
            pass

        return my_p

    def is_someone(self):
        # check for a new detection in vision topic
        # check if any faces have been detected
        if self.faces is not None and len(self.faces) > 0:
            return True
        else:
            return False

    def parser(self, command):
        rospy.loginfo("Receptionist is parsing")
        rospy.logwarn(self.inputText)
        time.sleep(1)
        intent = ""
        try:
            rospy.logwarn(command + self.inputText)
            intent = self.callGPT((command + " " + self.inputText))
            
            rospy.loginfo("Intent: " + intent)
            
        except Exception as e:
            rospy.logerr(e)
            self.say(promts["sorry" ])
            rospy.logdebug("Failed response")
        rospy.logwarn(intent)
        
        return intent
    

    def input_callback(self, msg):
        self.inputText = msg.inputText
        rospy.logdebug("I heard: " + self.inputText)
            
    def callGPT(self, pr, t_max=256):
        # rospy.logwarn("**************I am parsing in GPT: " + "'" + pr + "'"  + "*****************")
        
        response = openai.Completion.create(
            model=self.GPT_model,
            prompt=pr,
            temperature=self.GPT_temperature,
            max_tokens=t_max,
            top_p=1,
            frequency_penalty=self.GPT_frequency_penalty,
            presence_penalty=self.GPT_presence_penalty, 
        )
        return response.choices[0].text

def main():
    rospy.init_node('GPSR')
    rospy.loginfo("General Purpose Service Robot Created task created.")
    detector = DoorDetector()
    detector.run()
    gpsr()


if __name__ == '__main__':
    main()
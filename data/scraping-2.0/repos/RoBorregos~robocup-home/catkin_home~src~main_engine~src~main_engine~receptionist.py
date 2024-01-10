#!/usr/bin/env python3
import rospy
import actionlib
import time
from std_msgs.msg import String, Bool
from action_selectors.msg import RawInput
from humanAnalyzer.msg import face_array
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib 

import os
import openai
import json

promts = {
    "start": "Let's start receptionist task",
    "wait": "I am waiting for someone",
    "self_intro": "Hello, I'm homie, your personal assistant.",
    "ask_name": "What's your name?",
    "ask_drink": "What is your favorite drink?",
    "repeat": "i think i didn't get that, can you repeat it?",
    "come" : "come with me ",
    "unreached": "I am sorry, I couldn't reach the living room but let me try again ",
    "sorry" : "I am sorry, I think i didn't get that",
    "assign": "Can you please sit on the chair in front of me?"
}

calls = {
    "get_name": "if there is nos name return ''get me the name of the person on the next message: ",
    "get_drink": "get me the name of the beverage on the next message: ",
    "describe": "describe a person with the next attributes: ",
    "confirm" : "tell me onlyTrue or False, the next message is a general confirmation, like ues, OK, got it    :",
    "reached": "tell me only True or False, the next message is a confirmation that you reached a place:",

}


json_path = "/home/kevin/Desktop/workspace/home/TMR2023/robocup-home/catkin_home/src/humanAnalyzer/src/scripts/identities.json"

INITIALIZATION_STATE = "init"
START_STATE = "start"
WAIT_STATE = "waiting for someone"
IDENTIFY_STATE = "identifying the person"
GETTING_DRINK= "getting his favorite drink"
GOING_STATE = "going somewhere"
ASSIGN_STATE = "assing a seat to someone"
RETURN_STATE = "returning to the reception"
END_STATE = "done"



class person(object):
    def __init__(self ):
        self.id = None
        self.name = None
        self.drink = None
        self.seat = None
        self.age = None
        self.gender = None
        self.race = None

        

class receptionist(object):
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.currentState =  INITIALIZATION_STATE
        self.iteration = 0

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        
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
        
       


        self.currentState = START_STATE
        # wait for the robot to be ready
        time.sleep(1)
        # start the main loop
        # 
        # Array of persons
        self.faces = None
        self.persons = []

        self.run()
        rospy.spin()

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
        
    def say_callback(self, data):
        self.saying = data.data
        rospy.logwarn("saying: " + str(self.saying))

    def face_callback(self, data):
        self.faces = data.faces

    def run(self):

        while not rospy.is_shutdown():
            rospy.loginfo("-----------------------------------------RUN-----------------------------------------")

            if self.currentState == START_STATE:
                self.speech_enable.publish(Bool(False))

                rospy.loginfo("Receptionist is ready to start")
                self.say(promts["start"])               
                self.currentState = WAIT_STATE

                
            elif self.currentState == WAIT_STATE:
                if self.is_someone():
                    self.say(promts["self_intro"])
                    self.currentState = IDENTIFY_STATE

                else:
                    self.say(promts["wait"])
        

                
            elif self.currentState == IDENTIFY_STATE:
                name = self.get_name()
                

                if name != "":
                    drink = self.get_drink()


                    p = self.get_attributes()
                    p.name = name
                    p.drink = drink
                    self.persons.append(p)
                    # if  name not in self.persons.name:
                            
                    self.currentState = GOING_STATE
                else:
                    self.currentState = IDENTIFY_STATE

            elif self.currentState == GOING_STATE:
                # rospy.loginfo("People detected:")
                # for person in self.persons:
                #     rospy.loginfo(person.name)
                #     rospy.loginfo(person.race)
                #     rospy.loginfo(person.gender)
                rospy.loginfo("Receptionist is going to the living room")
                self.say(promts["come"] + name)
                go = self.moveTo(1)
                if go:
                    self.currentState = ASSIGN_STATE
                else:
                    self.currentState = GOING_STATE
                    self.say(promts["unreached"])
        
            elif self.currentState == ASSIGN_STATE:
                self.say("Hey" + name + promts["assign"])
                self.speech_enable.publish(Bool(True))
                intent = self.parser(calls["confirm"])
                if intent == "True":
                    self.currentState = RETURN_STATE
                intent = bool(intent)
                rospy.logwarn("intent: ")
                rospy.logwarn(intent)
            
                if intent == True:
                    if self.iteration == 1:
                        #self.introduce()
                        self.describe()
                    self.currentState = RETURN_STATE
                else:
                    self.currentState = ASSIGN_STATE
            
            elif self.currentState == RETURN_STATE:
                self.moveTo(0)
                self.iteration += 1
                if self.iteration == 2:
                    self.currentState = END_STATE
                else:
                    self.currentState = WAIT_STATE

            elif self.currentState == END_STATE:
                self.say(promts["end"])

                
                


    def say(self, text):
        time.sleep(1)   
        if self.saying == False:
            self.saying == True
            self.say_publisher.publish(text)
        time.sleep(len(text)/8 )
    
    def describe(self):
        rospy.logwarn("_____________describe_______________")
        for person in self.persons:
            if person.name !="":
                rospy.logwarn(person.name)
                rospy.logwarn(person.drink)
                rospy.logwarn(person.race)
        # self.say(promts["describe"])
        intent = self.parser(calls["describe"] + " " + str(self.persons[0].name) + " " + str(self.persons[0].race) + " " + str(self.persons[0].age))
        self.say(intent)
        
        rospy.logwarn("describe")
        time.sleep(1)

    def introduce(self):
        for person in self.persons:
            if person.name !="":
                rospy.logwarn(person.name)
                rospy.logwarn(person.drink)
                rospy.logwarn(person.race)

        intent = self.parser(calls["introdyce"] + " " + str(self.persons[0].name) + " to " + str(self.persons[0].race) + " " + str(self.persons[0].age))

        pass

    def get_name(self):
        self.speech_enable.publish(Bool(False))

        name = ""
        self.inputText = ""
        # get the name from the user
        rospy.loginfo("Receptionist is getting name")
        i = 0
        # wait for the user to say something
        rospy.logwarn("Waiting for user input")
        self.say(promts["ask_name"])
        self.speech_enable.publish(Bool(True))
        rospy.logwarn("name1: " + name)
        rospy.logwarn(name  != "" and not rospy.is_shutdown())
        while not rospy.is_shutdown() and name == "":
            i+=1
            time.sleep(1)
            rospy.logdebug(name)
            if self.inputText != "":
                    name  = self.parser(calls["get_name"])
            if i%28 == 0:
                self.speech_enable.publish(Bool(False))
                self.say(promts["ask_name"])
                self.speech_enable.publish(Bool(True))
                i = 0
                
        rospy.logwarn("name2: " + name)

        self.speech_enable.publish(Bool(False))

        rospy.logwarn(name)

        time.sleep(3)
        return name.rstrip('\n')
    
    def get_drink(self):
        self.inputText = ""
        drink = ""
        i = 0
        rospy.loginfo("Receptionist is getting drink")
        self.say(promts["ask_drink"])
        self.speech_enable.publish(Bool(True))
        while not rospy.is_shutdown() and drink == "":
            i += 1
            time.sleep(1)
            if self.inputText != "":
                drink  = self.parser(calls["get_drink"])
                rospy.logwarn("drink: " + drink)

            if i%28 == 0:
                self.speech_enable.publish(Bool(False))
                self.say(promts["ask_drink"])
                self.speech_enable.publish(Bool(True))
                i = 0
        # rospy.logwarn("drink: " + drink)
        self.speech_enable.publish(Bool(False))
        return drink.rstrip('\n')




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
    rospy.init_node('receptionist', anonymous=True)
    rospy.loginfo("Receptionist task created.")
    receptionist()


if __name__ == '__main__':
    main()
#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import speech_recognition as sr
from geometry_msgs.msg import Twist
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
client = OpenAI(api_key)

motion_command = Twist()
motion_publisher = None  
text_publisher = None  

def speech_to_text_callback(event):
	global text_publisher  # Declare text_publisher as global
	
	r = sr.Recognizer()
	#r.grammar = text_grammar
	r.energy_threshold = 4000
	# dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
	r.dynamic_energy_threshold = False
	
	try:
		with sr.Microphone() as source:
			print("Listening now: ")
			audio = r.listen(source, timeout=2)
			print("Stopped Listening")
			text = r.recognize_google(audio, show_all=True)
			print(text)
		
		alternative_list = text.get('alternative', [])
		
		# Iterating to find text with numeric digits
		selected_text = ""
		for item in alternative_list:
			transcript = item.get('transcript', '')
			if any(char.isdigit() for char in transcript):
				selected_text = transcript
				break
		
		# If no text with numeric digits found, select the first one
		if selected_text is None and alternative_list:
			selected_text = alternative_list[0].get('transcript', '')
		
		print("Selected Text:", selected_text)
		
		text_publisher.publish(selected_text)  # Publishing the recognized text
	
	except sr.UnknownValueError:
		rospy.logwarn("Could not recognize speech")
	
	except Exception as e:
		rospy.logerr("Speech recognition error: ", e)
	
	# rospy.sleep(2)  

def process_voice_command(text_msg):
	global motion_command, motion_publisher
	
	text = text_msg.data
	digit_match = re.search(r'\b([1-9]|10|1[1-9]|20|30)\b', text)
	
	# digit mapping approach (tedious)
	if digit_match:
		digit = int(digit_match.group(0))
		linear_value = float(digit) / 10.0
		rospy.loginfo("Recognized digit: %s", digit)
	else:
		linear_value = 0.0
	
	if "left" in text:
		rospy.loginfo("Command: Left")
		motion_command.angular.z = 0.2
		motion_command.linear.x = linear_value
		motion_publisher.publish(motion_command)
	
	elif "right" in text:
		rospy.loginfo("Command: Right")
		motion_command.angular.z = -0.2
		motion_command.linear.x = linear_value
		motion_publisher.publish(motion_command)
	
	elif "straight" in text:
		rospy.loginfo("Command: Straight")
		motion_command.angular.z = 0.0
		motion_command.linear.x = linear_value
		motion_publisher.publish(motion_command)
	
	elif "stop" in text:
		rospy.loginfo("Command: Stop")
		stop_robot()
	
	else:
		rospy.loginfo("Unrecognized command")
		stop_robot()  # Stop the robot for unrecognized commands
	
	# Additional check for command not matching the grammar
	if not re.match(text_grammar, text):
		rospy.logerr("Command does not match the grammar: ", text)
		stop_robot()


def stop_robot():
	global motion_command, motion_publisher
	motion_command.linear.x = 0.0
	motion_command.angular.z = 0.0
	motion_publisher.publish(motion_command)  # Publishing stop command

if __name__ == '__main__':
	text_grammar = """
	# Command format: "Move <direction> by <distance> units"
	# e.g., "Move left by ten units", "Move right by five units", "Turn left", "Turn right", "Stop", "Move straight"
	direction = "left" | "right" | "straight"
	distance = /([1-9]|10|1[1-9]|20|30)/  # Matches numbers from 1 to 30
	units = "units"
	command = ("move" direction "by" distance units | "turn" direction | "move" direction | "stop" | "move" direction )
	"""
	
	rospy.init_node('voice_commands2', anonymous=True)
	text_publisher = rospy.Publisher( '/recognized_text', String, queue_size=1)  # Publishing text
	
	# Creating a timer 
	timer = rospy.Timer(rospy.Duration(5), speech_to_text_callback)
	
	text_subscriber = rospy.Subscriber('/recognized_text',String, process_voice_command)
	motion_publisher = rospy.Publisher( '/cmd_vel', Twist, queue_size=1)  # Publishing movement commands
	
	rospy.spin()

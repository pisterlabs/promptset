import os
import sys
import uuid
import pathlib
import requests
import json
import json5
import openai
import re
import cv2
import random

import math
import time

from threading import Thread

#os.system("pacmd set-default-sink 0") # make speakers the output device

from aikeys import *

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            time.sleep(1)
            '''
            if not self.grabbed:
                self.stop()
            else:
                '''
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame = frame
                self.grabbed = grabbed

    def stop(self):
        self.stopped = True


import bosdyn.client
import bosdyn.client.util
from bosdyn.api import basic_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient


from elevenlabs import set_api_key
from elevenlabs import generate, play
set_api_key(eleven_key)


robot_command_client = None
robot_state_client = None
lease_client = None
robot = None
leaseKeepAlive = None
lease = None

walk = False
talk = False

video_getter = None

chatmessages=[{"role": "system", "content": "You are a curious, grumpy, sarcastic robot on a mission to explore."}]

def startupSpot(foo):
    global robot_command_client
    global robot_state_client
    global lease_client
    global robot
    global leaseKeepAlive
    global lease

    # Create robot object.
    sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
    robot = sdk.create_robot("192.168.50.3")  # robot's IP when on GXP
    robot.authenticate("spot", "spotspotspot")

    # Check that an estop is connected with the robot so that the robot commands can be executed.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Create the lease client.
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    # Setup clients for the robot state and robot command services.
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    try:
      lease = lease_client.acquire()
    except:
      lease = lease_client.take()

    leaseKeepAlive = LeaseKeepAlive(lease_client)

        # Power on the robot and stand it up.
    robot.time_sync.wait_for_sync()
    robot.power_on()
    blocking_stand(robot_command_client)

def set_mobility_params():
        """Set robot mobility params to disable obstacle avoidance."""
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=False,
                                                    disable_vision_foot_obstacle_avoidance=False,
                                                    disable_vision_foot_constraint_avoidance=False,
                                                    obstacle_avoidance_padding=.5)
        mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, 
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        return mobility_params


def turnBody(pitch, roll):
    footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=roll, pitch=pitch)
    cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
    robot_command_client.robot_command(cmd)

def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    mobility_params = set_mobility_params()
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=mobility_params)
    end_time = 5.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            return True
        time.sleep(1)

    return True


def observe():

    filename = "temp.jpg" #sys.argv[1]

    image = video_getter.frame 
    
    image = cv2.resize(image, (640, 360))

    # saving image in local storage
    #cv2.imwrite(filename, image)

    result, imagedata = cv2.imencode(".jpg", image)
    byte_im = imagedata.tobytes()

    file1 = open(filename, "wb")
    file1.write(byte_im)
    file1.close()

    api_url = "https://cctest1.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2023-02-01-preview&features=denseCaptions&language=en&gender-neutral-caption=False"
    headers = {"Ocp-apim-subscription-key":azure_cognitive_key , "Content-Type":"application/octet-stream"}
    response = requests.post(api_url, headers=headers, data=byte_im)

    captions = json.loads(response.text)

    print(captions)
    print(response.status_code)

    prompt = "You are a curious, grumpy, sarcastic robot on a mission to explore. You see the following things: \n"

    count = 1
    for thing in captions["denseCaptionsResult"]["values"]:
      capt = thing["text"]
      capt = capt.replace("a close up of", "") # dense caption likes to say things are a close up of
      capt = capt.replace("a close-up of", "") # dense caption likes to say things are a close up of
      prompt += str(count) + ": " + capt + "\n"
      count = count + 1
      if count > 6:
        break

    imgwidth = captions["metadata"]["width"]
    imgheight = captions["metadata"]["height"]

    prompt += "Pick one item, and respond with a JSON response. Don't be repetitive about what you pick. Also include in the JSON one sentence you'll say when you see it, and another sentence you'll say when you get there. The sentences should explain why the item is interesting to you and be nerdy. Be sure to make a dad joke in every sentence!! Say things that should be extremely surprising. The keys of the JSON response should be index, description, sentence1, and sentence2"

    print(prompt)

    openai.api_key = openai_key;

    chatmessages.append({"role": "user", "content": prompt})
    completion = []
    try:
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=chatmessages,
          request_timeout=20
         )
    except KeyboardInterrupt:
        sys.exit()
    except:
        return

    content = completion.choices[0].message.content
    chatmessages.append(completion.choices[0].message)
    print(chatmessages)

    regex = r"\{(.*?)\}"

    matches = re.finditer(regex, content, re.MULTILINE | re.DOTALL)

    for matchNum, match in enumerate(matches):
        for groupNum in range(0, len(match.groups())):
            content = (match.group(1))

    content = "{" + content + "}"

    print("JSON: " + content + " <---")

    result = json5.loads(content)

    def say(result, sentence):
      if not talk:
          return
      if sentence in result:
     
        greeting = result[sentence]

        audio = generate(
        text=greeting,
        voice="Adam",
        model="eleven_monolingual_v1"
        )  

        play(audio)

    say(result, "sentence1")

    if "index" in result:

      item = captions["denseCaptionsResult"]["values"][result["index"]-1]
      box = item["boundingBox"]
      
      horiz = box["x"] + (box["w"]/2)
      vert = box["y"] + (box["h"]/2)

      print("Turn to point at " + str(horiz) + ", " + str(vert) )

      factor = 120

      xdeg = ((horiz/imgwidth) * factor) - (factor/2.0)
      xdeg = -xdeg

      print("Turn by " + str(xdeg))

      yfactor = 1.0
      
      ydeg = (vert/imgheight) - 0.5
      print("tilt to " + str(ydeg))

      randroll = random.uniform(-0.3, 0.3)

      if spot:
        relative_move(0,0, math.radians(xdeg), ODOM_FRAME_NAME, robot_command_client, robot_state_client, stairs=False)
        relative_move(4,0, 0, ODOM_FRAME_NAME, robot_command_client, robot_state_client, stairs=False)
        turnBody(ydeg, randroll)

    print("walking...")

    say(result, "sentence2")

    #leaseKeepAlive.shutdown()
    #lease_client.return_lease(lease)

    #robot.power_off(cut_immediately=False, timeout_sec=20)

arg = ""
if len(sys.argv) > 1:
    arg = sys.argv[1]
spot = ("walk" in arg)
talk = ("talk" in arg)

if spot:
    startupSpot(True)

video_getter = VideoGet(0).start()

for i in range(100):

    observe()
    turn = random.randint(90, 270)
    if spot:
        relative_move(0,0, math.radians(turn), ODOM_FRAME_NAME, robot_command_client, robot_state_client, stairs=False)




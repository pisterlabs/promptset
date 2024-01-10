# Raphael Fortuna (raf269) 
# Rabail Makhdoom (rm857) 
# Final Project Report
# Lab 403, Lab Section:  4:30pm-7:30pm on Thursdays 

# main file to control the robot
import time
import openai
from importlib.machinery import SourceFileLoader

prePath = "/home/pi/"

directPaths = prePath + "RoboGPT/src/"
AICorePaths = prePath + "RoboGPT/src/AI_src/"

# import libraries
ultrasonicClass = SourceFileLoader("frontUltrasonic", directPaths + "ultrasonicClass.py").load_module()
motor_control = SourceFileLoader("motorControl", directPaths + "motor_control.py").load_module()
accelerometerClass = SourceFileLoader("accelerometerClass", directPaths + "accelerometerClass.py").load_module()
robot_demo = SourceFileLoader("robot_core", AICorePaths + "robot_core.py").load_module()
pin_config = SourceFileLoader("pin_config", directPaths + "pin_config.py").load_module()
fifo = SourceFileLoader("displayFifo", directPaths + "fifo.py").load_module()

try:
    speech_to_text = SourceFileLoader("speech_to_text", AICorePaths + "speech_to_text.py").load_module()    
except:
    # need the pi version
    speech_to_text = SourceFileLoader("speech_to_text", AICorePaths + "rpi_speech_to_text.py").load_module()

############################################################################################################

# if true, then take in text input instead of speech
toggleTextControl = False

# if true, then the robot is estopped
estopActive = False

# if true, then the robot just resumed from being estopped
justResumedFromEstop = False

def GPIO22_callback(channel):
    """ speech and text swap button callback"""
    
    global toggleTextControl
    toggleTextControl = not toggleTextControl

    if (toggleTextControl):
        print("Text input enabled")
    else:
        print("Speech input enabled")

def GPIO23_callback(channel):
    """ callback for the resume button """

    global estopActive
    global justResumedFromEstop
    justResumedFromEstop = True
    estopActive = False
    print("Estop deactivated")

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM) # Set for GPIO numbering not pin numbers
GPIO.setup(pin_config.text_swap_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(pin_config.resume, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(pin_config.text_swap_button, GPIO.FALLING, callback=GPIO22_callback, bouncetime=300)
GPIO.add_event_detect(pin_config.resume, GPIO.FALLING, callback=GPIO23_callback, bouncetime=300)

###########################################################################################################

# minimum distance for ultrasonic to be considered safe
safeDistance = 10 # cm

metersPerSecond = 0.1524 # m/s
degreesPerSecond = 69.23 # degrees/s (90/1.3)

class manager:
    """ class to manage the robot """

    def __init__(self, voice_on = True, clear_on_estop = False, ultrasonic_on = True):
        self.front_sensor = ultrasonicClass.frontUltrasonic()
        self.back_sensor = ultrasonicClass.backUltrasonic()
        self.motors = motor_control.motorControl()
        self.accelerometer = accelerometerClass.accelerometerClass()
        self.voice_on = voice_on
        self.core = robot_demo.robot_core(voice_on=voice_on)
        self.fifo = fifo.displayFifo(pin_config.fifo_path)
        self.audio_in = speech_to_text.speech_to_text(self.fifo.sendCommand)

        # sensor data
        self.forwardDistance = 0
        self.backwardDistance = 0

        # not all are sent to the core, but are left for future use if needed
        self.accelX = 0
        self.accelY = 0
        self.accelZ = 0

        self.gyroX = 0
        self.gyroY = 0
        self.gyroZ = 0

        # queue of actions to do
        self.actionQueue = []
        
        # what the robot is currently doing
        self.currentAction = None

        # if the robot did something
        self.didSomething = False

        # if true, prints out the sensor data and time it took to get it
        self.ultrasonic_on = ultrasonic_on

        # if true, then clear the action queue when estopped
        self.clear_on_estop = clear_on_estop

        self.refreshSensorData()

        print("Robot systems initialized")

    def refreshSensorData(self):
        """ refresh the sensor data with new data """
        startTime = time.time()

        self.forwardDistance = self.front_sensor.getDistanceNoWhile()
        self.backwardDistance = self.back_sensor.getDistanceNoWhile()

        if(self.ultrasonic_on):
            print("Took " + str(time.time() - startTime) + " seconds to get ultrasonic data")
            print("Forward distance: " + str(self.forwardDistance))
            print("Backward distance: " + str(self.backwardDistance))

        self.accelX, self.accelY, self.accelZ = self.accelerometer.getAcceleration()
        self.gyroX, self.gyroY, self.gyroZ = self.accelerometer.getGyro()

    def exportSensorData(self):
        """ export the sensor data as a dictionary """

        # create the dictionary
        sensorData = {}

        # add the ultrasonic data
        sensorData['ultrasonic_front'] = self.forwardDistance
        sensorData['ultrasonic_back'] = self.backwardDistance

        # add the accelerometer data
        sensorData['accelerometer'] = self.accelX
        # sensorData['accelerometer_y'] = self.accelY
        # sensorData['accelerometer_z'] = self.accelZ

        # add the gyro data, only want yaw since on a smooth surface
        sensorData['gyro'] = self.gyroZ

        return sensorData
    
    def sendSensorData(self):
        """ send the sensor data to the core """

        sensorData = self.exportSensorData()

        # in case runs out of memory
        try:
            self.core.update_current_sensor_data(str(sensorData))

        except openai.InvalidRequestError as e:
            # ran out of space, so need to reboot the core, wipes current memory
            self.reboot()

        except openai.APIError as e:
            print('OpenAI API is overloaded')
            print(e)
            self.fifo.sendCommand('OpenAI API is, overloaded sensor, data not sent')
            time.sleep(3)

    def listen(self):
        """ listen for speech """

        if toggleTextControl:
            # means should take in input from keyboard
            text = input("What you want to tell the robot: ")
            return text
        else:
            if self.audio_in.get_speech_text():
                # if successfully heard something, return it
                return self.audio_in.get_collected_text()
        
        # if there is no text, return an empty string
        print("Nothing was heard")
        return ""

    def reboot(self):
        # ran out of space, so need to reboot the core, wipes current memory
        print("########## ROBOT CORE IS REBOOTING ########")
        if self.voice_on:
            self.core.text_instance.speak_text("Oh no, I have run out of memory, I need to reboot my core")
        else:
            print("Oh no, I have run out of memory, I need to reboot my core")
        time.sleep(1)
        print("...")
        time.sleep(.5)
        print("...")
        time.sleep(.25)
        print("...")
        self.core.reboot_core()
        if self.voice_on:
            self.core.text_instance.speak_text("I am back online!")
        else:
            print("I am back online!")
        print("########## ROBOT CORE REBOOT COMPLETE ########")
    
    def process(self, text):
        """ process the text """

        if text != "":
            # got a command, so send it to ChatGPT

            try:
                response, commands = self.core.run_one_cycle(text)
                return response, commands
            
            except openai.InvalidRequestError as e:
                self.reboot()

        return "", []

    def addCommands(self, commands):
        """ add the commands to the action queue 
        
        command format is [{'moveBackward': '0.10'}, {'turnRight': '90'}, {'moveForward': '1'}]
        
        """
        for command in commands:
            self.actionQueue.append(command)

    def EStop(self):
        """ stop everything """

        self.motors.stop()

        if self.clear_on_estop:
            # no commands to do
            self.actionQueue = []

            # no current action
            self.currentAction = None
        else:
            # pause the current action
            try:
                if self.currentAction is not None or not self.currentAction.isDone():
                    self.currentAction.pausedTime = time.time()
            except:
                print("Failed to modify current action")

        # update that the robot is stopped
        self.refreshSensorData()

        global estopActive
        estopActive = True

        # send the command to show the resume arrow
        self.fifo.sendCommand(pin_config.resume_fifo)

        print("EStop Activated")

    def reactToSensors(self):
        """ poll the sensors and react to them """
        # print("Refreshing Sensor Data")
        self.refreshSensorData()

        # if the front sensor is too close, stop
        if self.forwardDistance < safeDistance:
            print("Too close to front")
            print("Forward Distance: " + str(self.forwardDistance))
            self.EStop()

        # if the back sensor is too close, stop
        if self.backwardDistance < safeDistance:
            print("Too close to back")
            print("Backward Distance: " + str(self.backwardDistance))
            self.EStop()
        
    def runRobot(self, actionName):
        """ runs the robot based off of a action name """

        # run the right function from motor control
        if actionName == 'moveForward':
            self.motors.moveStraight()
            return True

        elif actionName == 'moveBackward':
            self.motors.moveBack()
            return True

        elif actionName == 'turnLeft':
            self.motors.turnLeft()
            return True

        elif actionName == 'turnRight':
            self.motors.turnRight()
            return True

        elif actionName == 'stop':
            self.motors.stop()
            return True

        else:
            print("Invalid Action Name")
            return False

    def getTimeConversion(self, actionName, actionValue):
        """ converts the action value to a time period """

        time_period = 0

        if actionName == 'moveForward' or actionName == 'moveBackward':
            # the action value is in meters, so convert to seconds
            time_period = float(actionValue) / metersPerSecond

        elif actionName == 'turnLeft' or actionName == 'turnRight':
            # the action value is in degrees, so convert to seconds
            time_period = float(actionValue) / degreesPerSecond

        return time_period

    def doAction(self):
        """ do an action in the queue """

        # only do an action if there is one to do
        if self.actionQueue != []:

            # get the first action
            action = self.actionQueue.pop(0)

            # get the action name
            actionName = list(action.keys())[0]

            # get the action value
            actionValue = action[actionName]

            if self.runRobot(actionName):

                # convert the action value to a time period
                try:
                    time_period = self.getTimeConversion(actionName, actionValue)
                except:
                    print("Invalid Command Value, stopping Robot")
                    self.motors.stop()
                    return

                self.currentAction = pin_config.actionUnit(time_period, actionName, actionValue)

    def restartPreviousAction(self):
        """ restart the previous action """
        try:
            if self.currentAction is not None or not self.currentAction.isDone():

                # start action
                self.runRobot(self.currentAction.action_name)

                # update the start time and the time passed
                self.currentAction.time_period = self.currentAction.pausedTime - self.currentAction.start_time
                self.currentAction.start_time = time.time()
        except:
            print("Current action could not be modified")
            self.motors.stop()

    def actionCycle(self):
        """ run the robot's actions once """

        # if the current action is done, do the next action - otherwise keep doing the current action
        # and monitor the sensors for any problems that might lead to a EStop
        if (self.currentAction == None or self.currentAction.isDone()):

            # stop the robot
            self.motors.stop()

            # run more actions
            if self.actionQueue != []:

                exportedActions = ""

                # send to be shown on display
                for i in self.actionQueue:
                    exportedActions += str(i) + ','
                
                self.fifo.sendCommand(exportedActions)
                self.didSomething = True
                self.doAction()

            # no more actions, get ready for the next command
            else:

                # send the sensor data to the core
                # this can take unknown amounts of time, so first it stops
                # update the user that the robot is stopped
                self.fifo.sendCommand("Robot Stopped")

                self.sendSensorData()

                # say that the robot is ready
                # let the robot know it is done moving
                if (self.didSomething):
                    self.core.update_current_sensor_data("The robot has finished moving, please let the user know.", speak = self.voice_on)

                # listen for speech
                text = self.listen()

                # process the text, response spoken by the robot inside of robot_core
                response, commands = self.process(text)

                # add the commands to the action queue
                self.addCommands(commands)
                self.didSomething = False
            
        # react to the sensors
        self.reactToSensors()


if __name__ == "__main__":
    # create the robot
    robot = manager(voice_on=True)
    print("Robot created")

    robot.core.intialize_core()

    # run the robot
    while True:
        if not estopActive:
            if justResumedFromEstop:
                justResumedFromEstop = False
                robot.restartPreviousAction()
            robot.actionCycle()
            




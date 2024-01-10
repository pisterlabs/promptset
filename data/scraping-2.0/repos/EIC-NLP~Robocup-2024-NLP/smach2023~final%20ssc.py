import subprocess
import smach
import nlp_client
from ratfin import *
import sys

from cv_connector.msg import CV_type

from cv_connector.srv import CV_srv, CV_srvRequest, CV_srvResponse
from dy_custom.srv import SetDigitalGripper, SetDigitalGripperRequest, SetDigitalGripperResponse, SetDegreeRequest,SetDegree,SetDegreeResponse
from core_nlp.utils import WakeWord , GetIntent
from core_smach.follow_person import Follow_Person
import signal
import roslib
import rospy
import smach
import smach_ros
import nlp_client
import threading
from ratfin import *
import time
import os
import threading
from dy_custom.srv import SetDigitalGripper, SetDigitalGripperRequest, SetDigitalGripperResponse, SetDegreeRequest,SetDegree,SetDegreeResponse
import openai

# Safety
from core_nlp.emerStop import EmergencyStop
# smach state locations
from core_nlp.get_time import TellTime, TellDay
from core_nlp.utils import WakeWord, Speak, GetEntities, GetIntent, GetName, GetObject, GetLocation
from core_smach.follow_person import Follow_Person
from task_gpsr_revised import GTFO, GoGo, IdleGeneral, SimGraspObjectGPSR, SimPlaceObjectGPSR, ExampleState
from ssc_og import State1, State2
""" 

List of State availiable for SSC later
------------------------
TellTime
TellDay
GTFO (ask him to leave)
GetName
SimGraspObjectGPSR 
SimPlaceObjectGPSR


"""



class SmachGeneratorState(smach.State):
    def __init__(self):
        """ input_keys=['state_sequence']
         userdata.state_sequence = ['State1', 'State2','State1'] """
        smach.State.__init__(self,
                             outcomes=['success', 'failure'],
                             input_keys=['state_sequence'],
                             output_keys=['execution_result'])
        ## RULES

        # Mapping of state names to classes
        state_mapping = {
            'State1': State1(),
            'State2': State2(),
            # Add more states here...
        }

        ###

    def generate_state_dict_list(self, state_sequence):
        state_dict_list = []
        
        for i in range(len(state_sequence)):
            current_state = state_sequence[i]
            state_obj = self.state_mapping[current_state]
            current_state_with_suffix = current_state + "_" + str(i + 1)  # appending the suffix

            # Define remappings here
            remappings = {'data1': 'data1', 'data2': 'data2'}  # Use actual remappings

            if i == len(state_sequence) - 1:  # Last state
                transitions = {'success': 'final_outcome', 
                            'failure': state_sequence[i - 1] + "_" + str(i) if i > 0 else 'final_outcome', 
                            'loop': current_state_with_suffix}
            else:
                transitions = {'success': state_sequence[i + 1] + "_" + str(i + 2), 
                            'failure': state_sequence[i - 1] + "_" + str(i) if i > 0 else state_sequence[i + 1] + "_" + str(i + 2), 
                            'loop': current_state_with_suffix}

            state_dict_list.append({
                'state_name': current_state_with_suffix,
                'state_obj': state_obj,
                'transitions': transitions,
                'remappings': remappings  # Added remappings here
            })

        return state_dict_list


    def construct_smach_state_machine(self, states_dict_list):
        sm = smach.StateMachine(outcomes=['final_outcome'])

        with sm:
            for state_dict in states_dict_list:
                smach.StateMachine.add(state_dict['state_name'], 
                                    state_dict['state_obj'],
                                    transitions=state_dict['transitions'])

        return sm
    
    def execute(self, userdata):
        state_sequence = userdata.state_sequence
        state_dict_list = self.generate_state_dict_list(state_sequence)
        sm = self.construct_smach_state_machine(state_dict_list)

        # Write to a file
        filename = "generated_smachFUCKKKKKKKKKKKK.py"
        with open(filename, 'w') as f:
            f.write("""
import rospy
import smach
import threading
from core_nlp.emerStop import EmergencyStop

NODE_NAME = "smach_task_gpsr_revised"
rospy.init_node(NODE_NAME)

# Create a SMACH state machine
sm = smach.StateMachine(outcomes=['out0'])

# Declare Variables for top-state
sm.userdata.name = ""
sm.userdata.intent = ""

            """)

            # Dynamically add states to smach
            for state in state_dict_list:
                f.write("""
with sm:
    smach.StateMachine.add('{0}',
                           {1},
                           remapping={2},
                           transitions={3})
                """.format(state['state_name'], state['state_obj'], state['remappings'], state['transitions']))

            # Add EmergencyStop code to the file
            f.write("""
# Create a thread to execute the smach container
smach_thread = threading.Thread(target=sm.execute)
smach_thread.start()

es = EmergencyStop()
es_thread = threading.Thread(target=es.execute)
es_thread.start()
es.emer_stop_handler()
            """)

        # Execute the generated script
        try:
            subprocess.run(["python3", filename], check=True)
            userdata.execution_result = "success"
            return 'success'
        except subprocess.CalledProcessError:
            userdata.execution_result = "failure"
            return 'failure'


def main():
    # Initialize the node
    NODE_NAME = "TestSmachGenerator"
    rospy.init_node(NODE_NAME)
    
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['out0'])
    # Declear Variables for top-state
    # sm.userdata.name = ""
    # sm.userdata.intent = ""
    sm.userdata.state_sequence = ['State1', 'State2','State1']
    with sm:

        smach.StateMachine.add('TEST_GENERATOR',
                            SmachGeneratorState(),
                            remapping={"state_sequence": "state_sequence"},
                            transitions={'success': 'out0',
                                        'failure': 'out0'}
                                        )
        
        
        
    

    # Create a thread to execute the smach container
    # Execute SMACH plan
    smach_thread = threading.Thread(target=sm.execute)
    smach_thread.start()

    es = EmergencyStop()
    es_thread = threading.Thread(target=es.execute)
    es_thread.start()
    es.emer_stop_handler()

    
if __name__ == '__main__':
    main()



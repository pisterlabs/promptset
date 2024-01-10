import os
import sys
import time
import requests
import json
import argparse
from typing import Dict, Any
import logging

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface, SimulatorState, SimulatorSessionResponse
from bonsai_gym import GymSimulator3

log = logging.getLogger("gym_simulator")
log.setLevel(logging.INFO)

# Runner global variables 
# "http://localhost:5000/v1/prediction"
URL = "http://localhost:5000"
PREDICTION_PATH = "/v1/prediction"
HEADERS = {
  "Content-Type": "application/json"
}

# Build the endpoint reference
ENDPOINT = URL + PREDICTION_PATH


class MujocoPendulum(GymSimulator3):

    # Environment name, from openai-gym
    environment_name = "InvertedDoublePendulum-v4"

    # simulator name from Inkling
    simulator_name = "Mujoco_InvertedDoublePendulum_Simulator"

    def __init__(self, args):

        config_client = BonsaiClientConfig(args)
        self.args = args

        render = False if self.args.headless else True
        log.info("Render MujocoPendulum is set to ", str(render) )

        super().__init__(config_client, iteration_limit = 0, skip_frame = 1, render = render)       
 

    def step(self, action) -> Dict[str, Any]:
        
        log.debug("Step with action: ", action )
        next_state, step_reward, done, truncated, info = self.gym_env.step(action['action']) 

        returned_dict = self.gym_to_state(next_state, step_reward, done)
        log.debug("Step returned_dict is: ", returned_dict )

        return {
            'sim_halted': False,
            'key': returned_dict
        }
    
    
    def gym_to_state(self, next_state ):
        
        try:
            if len(next_state) == 2 and type(next_state[1])==dict:
                next_state = next_state[0]

        except BaseException as err:
            log.critical(f"Runner stopped gym_to_state does not received the correct states: {type(err).__name__}: {err}")  
                
        state = {
            "pos": float(next_state[0]),
            "sin_hinge1": float(next_state[1]),
            "sin_hinge2": float(next_state[2]),
            "cos_hinge1": float(next_state[3]),
            "cos_hinge2": float(next_state[4]),
            "velocity": float(next_state[5]),
            "ang_velocity1": float(next_state[6]),
            "ang_velocity2": float(next_state[7]),
            "constraint1": float(next_state[8]),
            "constraint2": float(next_state[9]),
            "constraint3": float(next_state[10])
        }

        return state


    def action_to_gym(self, brain_action ):
        
        log.debug("Brain_action is: ", brain_action )
        action = [brain_action['action'][0]]

        return action        



def train(args):

    sim_model = MujocoPendulum(args)
    sim_model.run_gym()


def run(args):

    sim_model = MujocoPendulum(args)
    sim_model_state = sim_model.reset()   

    try:
        while True:

            # Send states to brain for getting actions
            # Send the POST request
            response = requests.post(
                        ENDPOINT,
                        data = json.dumps(sim_model_state['key']),
                        headers = HEADERS
                    )

            # Extract the JSON response
            prediction = response.json()
            # Access the JSON result and set action
            action = {'action': [prediction['input_force']]}

            #Send actions to sim for next step
            sim_model_state = sim_model.step(action)

            #Stop the run if mujoco is returning terminal is True
            if sim_model_state['key']["_gym_terminal"] ==1.0:
                break


    except BaseException as err:
        log.critical(f"Runner stopped because {type(err).__name__}: {err}")        

    
def parse_kw_args(args):
    """
    Parse keyword args into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval    


def main(argv):

    parser = argparse.ArgumentParser(exit_on_error=False)
             
    parser.add_argument(
        '--debug', help='debug.', action='store_true')       
    parser.add_argument(
        '--run', help='run with docker.', action='store_true')     
    parser.add_argument(
        '--headless', help='Render', action='store_true')    

    args = parser.parse_args()  

    if args.debug: 
        log.setLevel(logging.DEBUG)

    if args.run:
        runner = run(args)
    else:
        trainer = train(args)

if __name__ == '__main__':

    #Init logging
    main(sys.argv)    
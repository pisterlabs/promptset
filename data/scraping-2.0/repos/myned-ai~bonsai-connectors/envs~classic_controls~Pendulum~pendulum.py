import logging
from typing import Any, Dict

import numpy as np
from gym_connectors import BonsaiConnector, GymSimulator

log = logging.getLogger("pendulum")

class Pendulum(GymSimulator):
    """ Implements the methods specific to Open AI Gym Pendulum environment 
    """    

    environment_name = 'Pendulum-v0'           # Environment name, from openai-gym

    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Pendulum environment
        """
        self.bonsai_state = {"cos_theta": 0.0,
                             "sin_theta": 0.0,
                             "angular_velocity": 0.0}

        super().__init__(iteration_limit, skip_frame)

    def gym_to_state(self, observation) -> Dict[str, Any]:
        """ Converts openai environment observation to Bonsai state, as defined in inkling
        """
        self.bonsai_state = {"cos_theta": float(observation[0]),
                             "sin_theta": float(observation[1]),
                             "angular_velocity": float(observation[2])}

        return self.bonsai_state

    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.       
        """
        actionValue = action['command']
        return [actionValue]      # Pendulum environment expects an array of actions 

    def gym_episode_start(self, config: Dict[str, Any]):
        """ Called during episode_start() to return the initial observation
            after reseting the gym environment. 

            config parameter is passed from an inkling lesson and can contain initial state
        """
        if config is None:
             config:Dict[str,Any] = {}

        super().gym_episode_start(config)

        #get the initial angle and angular velocity from config, 
        # or if not passed in, use the value set with env.reset() 
        initial_theta = config.get(
            "initial_theta", self._env.unwrapped.state[0])
        initial_angular_velocity = config.get(
            "initial_angular_velocity", self._env.unwrapped.state[1])

        # set the environment state
        self._env.unwrapped.state = np.array(
            [initial_theta, initial_angular_velocity])

        # set the bonsai state
        self.bonsai_state = {"cos_theta": float(np.cos(initial_theta)),
                             "sin_theta": float(np.sin(initial_theta)),
                             "angular_velocity": float(initial_angular_velocity)}

        # return the initial observation
        return np.array([np.cos(initial_theta), np.sin(initial_theta), initial_angular_velocity])

    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment 
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state


if __name__ == "__main__":
    """ Creates a Pendulum environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("pendulum")
    log.setLevel(level='DEBUG')

    #if more information is needed, uncomment this
    #gymlog = logging.getLogger("GymSimulator")
    #gymlog.setLevel(level='DEBUG')

    pendulum = Pendulum()
    connector = BonsaiConnector(pendulum)

    while connector.run():
        continue

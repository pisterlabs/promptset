import logging
from typing import Any, Dict


from gym_connectors import BonsaiConnector, PyBulletSimulator

log = logging.getLogger("reacher")


class Reacher(PyBulletSimulator):
    """ Implements the methods specific to Hopper environment
    """

    environment_name = 'ReacherPyBulletEnv-v0'  # Environment name, from openai-gym

    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Reacher environment
        """
        self.prev_potential : float = None

        self.bonsai_state = None

        super().__init__(iteration_limit, skip_frame)

    def gym_to_state(self, observation) -> Dict[str, Any]:
        """ Converts openai environment state to Bonsai state, as defined in inkling
        """
        potential = float(self._env.unwrapped.potential)
        if self.prev_potential is None:
            self.prev_potential = potential

        progress = potential - self.prev_potential
        
        self.bonsai_state = {"target_x": float(observation[0]),
                             "target_y": float(observation[1]),
                             "to_target_x": float(observation[2]),
                             "to_target_y": float(observation[3]),
                             "cos_theta": float(observation[4]),
                             "sin_theta": float(observation[5]),
                             "theta_velocity": float(observation[6]),
                             "gama": float(observation[7]),
                             "gama_velocity": float(observation[8]),
                             "rew": self.get_last_reward(),
                             "episode_rew": self.get_episode_reward(),
                             "progress": progress}
        
        self.prev_potential = potential
        
        return self.bonsai_state

    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.
        """
        central_joint_torque = action['central_joint_torque']
        elbow_joint_torque = action['elbow_joint_torque']

        # Reacher environment expects an array of actions
        return [central_joint_torque, elbow_joint_torque]

    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state

    def episode_start(self, config: Dict[str, Any] = None) -> None:
        """Reset the prev_potential at the beginning of each episode
        """
        self.prev_potential = None    

        super().episode_start(config)

    def initialize_camera(self,distance, yaw, pitch, x=0,y=0,z=0):
        """Initializes the position of Camera
        """
        lookat = [x, y, z]

        self._env.unwrapped._p.resetDebugVisualizerCamera(
            distance, yaw, pitch, lookat)

if __name__ == "__main__":
    """ Creates a Reacher environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("reacher")
    log.setLevel(level='INFO')

    # if more information is needed, uncomment this
    # gymlog = logging.getLogger("GymSimulator")
    # gymlog.setLevel(level='DEBUG')

    reacher = Reacher()
    connector = BonsaiConnector(reacher)

    while connector.run():
        continue

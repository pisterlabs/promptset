import sys
import logging
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from bonsai_gym import GymSimulator3

log = logging.getLogger("gym_simulator")
log.setLevel(logging.DEBUG)


class Taxi(GymSimulator3):
    # Environment name, from openai-gym
    environment_name = "Taxi-v2"

    # simulator name from Inkling
    simulator_name = "taxi_simulator"

    # convert openai gym observation to our state type
    def gym_to_state(self, observation):
        state = {"location": int(observation)}
        return state

    # convert our action type into openai gym action
    def action_to_gym(self, action):
        return action["command"]


if __name__ == "__main__":
    # create a brain, openai-gym environment, and simulator
    config = BonsaiClientConfig(argv=sys.argv)
    sim = Taxi(config, iteration_limit=200)
    sim.run_gym()

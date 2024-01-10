import sys
import logging
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from bonsai_gym import GymSimulator3

log = logging.getLogger("gym_simulator")
log.setLevel(logging.DEBUG)


class TapeCopy(GymSimulator3):
    # Environment name, from openai-gym
    environment_name = "Copy-v0"
    simulator_name = "tapecopy_simulator"

    # convert openai gym observation to our state type
    def gym_to_state(self, observation):
        state = {"character": observation}
        return state

    # convert our action type into openai gym action
    def action_to_gym(self, actions):
        return [actions["move"], actions["write"], actions["char"]]


if __name__ == "__main__":
    # create a brain, openai-gym environment, and simulator
    config = BonsaiClientConfig(argv=sys.argv)
    sim = TapeCopy(config)
    sim.run_gym()

import argparse


from tensorflow_agents.mountain_car_model_tester import MountainCarModelTester
from tensorflow_agents.mountain_car_mo_dqn import MultiObjectiveMountainCarDQN
from tensorflow_agents.mountain_car_mo_ddqn import MultiObjectiveMountainCarDDQN
from tensorflow_agents.mountain_car_mo_pddqn import MultiObjectiveMountainCarPDDQN
from tensorflow_agents.mountain_car_mo_wpddqn import MultiObjectiveWMountainCar
from tensorflow_agents.mountain_car_graphical_ddqn import MountainCarGraphicalDDQN
from tensorflow_agents.mountain_car_open_ai import OpenAIMountainCar

from tensorflow_agents.deep_sea_baseline_dqn import DeepSeaTreasureBaselineDQN
from tensorflow_agents.deep_sea_baseline_ddqn import DeepSeaTreasureBaselineDDQN
from tensorflow_agents.deep_sea_graphical_pddqn import DeepSeaTreasureGraphicalPDDQN
from tensorflow_agents.deep_sea_graphical_ddqn import DeepSeaTreasureGraphicalDDQN
from tensorflow_agents.deep_sea_graphical_dqn import DeepSeaTreasureGraphicalDQN
from tensorflow_agents.deep_sea_mo_wdqn import DeepSeaWAgent
from tensorflow_agents.deep_sea_graphical_wpddqn import MultiObjectiveDeepSeaW

from tensorflow_agents.mario_baseline import MarioBaseline

parser = argparse.ArgumentParser(description='Run agentArg model for game')
parser.add_argument("-a", "--agentArg", required=True)

args = parser.parse_args()
agentArg = args.agentArg

if agentArg == 'mountain_car_mo_dqn':
  agent = MultiObjectiveMountainCarDQN(1001)
elif agentArg == 'mountain_car_mo_ddqn':
  agent = MultiObjectiveMountainCarDDQN(1001)
elif agentArg == 'mountain_car_mo_pddqn':
  agent = MultiObjectiveMountainCarPDDQN(1001)
elif agentArg == 'mountain_car_mo_wpddqn':
  agent = MultiObjectiveWMountainCar(5000)
elif agentArg == 'mountain_car_graphical_ddqn':
  agent = MountainCarGraphicalDDQN(5000)
elif agentArg == 'mountain_car_open_ai':
  agent = OpenAIMountainCar(2000)
elif agentArg == 'deep_sea_baseline_ddqn':
  agent = DeepSeaTreasureBaselineDDQN(350)
elif agentArg == 'deep_sea_graphical_pddqn':
  agent = DeepSeaTreasureGraphicalPDDQN(301)
elif agentArg == 'deep_sea_baseline_dqn':
  agent = DeepSeaTreasureBaselineDQN(300)
elif agentArg == 'deep_sea_mo_wdqn':
  agent = DeepSeaWAgent(2000)
elif agentArg == 'deep_sea_graphical_ddqn':
  agent = DeepSeaTreasureGraphicalDDQN(1501)
elif agentArg == 'deep_sea_graphical_dqn':
  agent = DeepSeaTreasureGraphicalDQN(2001)
elif agentArg == 'deep_sea_graphical_wpddqn':
  agent = MultiObjectiveDeepSeaW(301)
elif agentArg == 'mario_baseline':
  agent = MarioBaseline(2000)

agent.train()

'''
agentArg = MountainCarModelTester("./mountain_car_wnet_54540.chkpt")
agentArg.test()
'''

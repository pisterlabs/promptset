# from MCTS_async import MCTS
from RRT import rrt
from exploration_simulator import Exploration
import openai
import os
if __name__ == '__main__':

    openai.api_key = "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" # Your openai key
    os.environ['OPENAI_API_KEY'] = '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'
    print(os.getenv("OPENAI_API_KEY"))
    ETA = 1 # Scaling factor for exploration
    GAMMA = 3 # N(S_t+1)^gamma
    K = 0.5  # Shapness of K
    d0 = 60 # This is the midpoint of sigmoid function, like a desired distance
    N = 3 # Action space and tree width
    L = 1  # Tree length
    fov = 86 # Field of View
    rom = 120 # Range of motion
    explorer_instance = None  # Declare this outside the try block so it can be accessed in the finally block

    """Examples prompt simulation"""
    # Level 1: Find X
    ################################
    find_fountain_goal = "Find me a Fountain"
    # Example NED position (N, E, D)
    find_fountain_start_pos = (-5, 0, -5) 
    find_fountain_start_yaw = 0
    #################################


    # Level 2: Find X conditioned on Y
    #################################
    find_pot_plant_on_table_goal = "Find me a pot plant on the table"
    # Example NED position (N, E, D)
    find_pot_plant_on_table_start_pos = (60, -40, -5) 
    find_pot_plant_on_table_start_yaw = 180
    #################################
    find_table_under_umbrella_goal = "Find me a table under the umbrella"
    # Example NED position (N, E, D)
    find_table_under_umbrella_start_pos = (60, 10, -5) 
    find_table_under_umbrella_start_yaw = 180



    # Level 3: Find X conditoned on path P
    #################################
    find_trash_but_on_pave_goal = "Find me a trashcan but stay on the pave way, don't go on the grass"
    # Example NED position (N, E, D)
    find_trash_but_on_pave_start_pos = (20, -45, -5) 
    find_trash_but_on_pave_start_yaw = 0
    #################################
    find_fire_hydrant_on_road_goal = "Find me a fire hydrant but stay on the road"
    # Example NED position (N, E, D)
    find_fire_hydrant_on_road_start_pos = (60, -40, -5) 
    find_fire_hydrant_on_road_start_yaw = 180
    #################################


    # Level 4: Find abstract A
    #################################
    find_eat_goal = "Find me a place to eat under the shade"
    # Example NED position (N, E, D)
    find_eat_start_pos = (60, -40, -5) 
    find_eat_start_yaw = 180
    #################################
    find_nap_goal = "Find me a good place to take a nap outside"
    # Example NED position (N, E, D)
    find_nap_start_pos = (-10, 0, -5) 
    find_nap_start_yaw = 0



    """Running Example"""
    exp_name = "Sim_DEMO_RRT_1_fountain"
    exp_type = "baseline" # choose baseline, RRT
    goal = find_fountain_goal
    start_pos = find_fountain_start_pos
    start_yaw = find_fountain_start_yaw

    try:
        rrt_instance = rrt(N, L, goal)
        print("RRT Instance established")
        # Initialize Exploration
        explorer_instance = Exploration(exp_name, exp_type, start_pos, start_yaw, rrt_instance, K, d0, N, fov, rom, goal)
        print("Explorer")
        # Run the exploration process
        explorer_instance.explore()

    except KeyboardInterrupt:
        print("\nReceived a KeyboardInterrupt! Stopping the exploration process.")
    finally:
        if explorer_instance:
            explorer_instance.cleanup() 
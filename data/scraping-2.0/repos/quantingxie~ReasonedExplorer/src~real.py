from RRT import rrt
from exploration_google_map import Exploration
import openai
import os
if __name__ == '__main__':

    
    openai.api_key = "" # Your openai API key
    MODEL = "gpt-3.5-turbo-16k"
    os.environ['OPENAI_API_KEY'] = '' # Your openai API key
    print(os.getenv("OPENAI_API_KEY"))
    ETA = 1 # Scaling factor for exploration
    GAMMA = 3 # N(S_t+1)^gamma
    K = 0.5  # Shapness of K
    d0 = 60 # This is the midpoint of sigmoid function, like a desired distance
    N = 3 # Action space and tree width
    L = 1  # Tree length
    fov = 120 # Field of View
    rom = 54 # Range of motion
    explorer_instance = None  # Declare this outside the try block so it can be accessed in the finally block



    exp_name = "Real_Exp_RRT_4"
    exp_type = "baseline" # baseline, RRT
    initial_gps = (40.4410146, -79.9444547)
    initial_yaw = 180

    try:
        goal = "Find a trash can."
        # Initialize MCTS
        rrt_instance = rrt(N, L, goal, MODEL)
        print("MCTS Instance established")
        # Initialize Exploration with the MCTS instance
        explorer_instance = Exploration(exp_name, exp_type, initial_gps, initial_yaw, rrt_instance, K, d0, N, fov, rom, goal, MODEL)
        print("Explorer")
        # Run the exploration process
        explorer_instance.explore()

    except KeyboardInterrupt:
        print("\nReceived a KeyboardInterrupt! Stopping the exploration process.")

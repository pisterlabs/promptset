from Children import Children
from secret_stuff import API_KEYS
from configuration import LAMBDA_
from openai_queries import experiment, improve_preprompt

def initialize_and_start_threads(data_list, prompt, iterations, sleep_time):
    children = []
    improved_preprompts = []
    for index, key in enumerate(API_KEYS):
        # create a prompt for each child
        improved_preprompts.append(improve_preprompt(key, prompt))
        children.append([improved_preprompts[index], Children(target=experiment, args=(data_list, key, improved_preprompts[index], iterations, sleep_time))])
        children[index][1].start()
        print(f'Child {index} prompt is :{improved_preprompts[index]}')
    return children

def calculate_score(X, Y, Y_max):
    lambda_ = LAMBDA_
    if Y_max == 0:
        Y_max = 1
    elif Y_max == None:
        return 0
    return X - lambda_ * (Y / Y_max)
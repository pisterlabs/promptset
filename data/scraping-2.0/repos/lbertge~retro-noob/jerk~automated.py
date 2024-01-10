import sys
import re
import time
import numpy as np
import json
import docker
import os
from subprocess import check_output, PIPE, call

EMA_RATE = 0.2
EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6)
MOVE_EPISODE = 100
BACKTRACK_EPISODE = 100
jump_prob = 5.0 / 100.0
jump_repeat = 3

def finished():
    print('Checking...')
    a = check_output(["retro-contest", "job", "show", "-v"], stdin=PIPE, stderr=PIPE)
    finish = re.search('ID:\s\d+\nStatus:\s(\w+)\n', a.decode("utf-8"))
    print(a)
    if finish.group(1) == "finished":
        return True
    elapsed_time = re.findall('ETA\s\(seconds\):\s(\d+\.?\d+)\n', a.decode("utf-8"))

    try:
        elapsed_time = list(map(float, elapsed_time))
        print(elapsed_time)
        max_time_remaining = max(elapsed_time)
        print(str(max_time_remaining) + 'approx. remaining')
        time.sleep(max_time_remaining)

    except:
        # wait 5 minutes
        print('I encountered an error, perhaps the time is undetermined?')
        time.sleep(300)
    return False

DOCKER_REGISTRY = os.environ['DOCKER_REGISTRY']

client = docker.from_env()

current_dir = os.getcwd()

for jump_repeat in [1, 2, 3, 4]:
    for jump_prob in [1.0, 5.0, 10.0, 15.0]:
        for EXPLOIT_BIAS in [0.10, 0.20, 0.30, 0.4]:
            fname = "jr%sjp%sep%s" % (str(jump_repeat), str(int(jump_prob)), str(EXPLOIT_BIAS))
            params = {
                    "jump_repeat": jump_repeat,
                    "jump_prob": jump_prob,
                    "exploit_bias": EXPLOIT_BIAS,
                    "ema_rate": EMA_RATE,
                    "move_episode": MOVE_EPISODE,
                    "backtrack_episode": BACKTRACK_EPISODE

            }
            with open(fname, 'w') as f:
                json.dump(params, f)

            with open('jerk_grid.docker', 'w') as f:
                f.write('FROM openai/retro-agent:bare\n')
                f.write('ADD jerk_grid_search.py ./agent.py\n')
                f.write('ADD %s ./params.json\n' %fname)
                f.write('CMD ["python", "-u", "/root/compo/agent.py"]')
            # with open('jerk_grid.docker', 'rb') as f:
                # client.images.build(path=os.getcwd(), fileobj=f, tag=str(DOCKER_REGISTRY + fname), dockerfile=os.getcwd()+'/jerk_grid.docker')
            agent_tag = 'jerk-grid:' + fname
            build = check_output(['docker', 'build', '-f', 'jerk_grid.docker', '-t', DOCKER_REGISTRY + '/' + agent_tag, '.'], stdin=PIPE, stderr=PIPE)
            push = check_output(['docker', 'push', DOCKER_REGISTRY + '/' + agent_tag])
            print('Finished build and push')

            for i in range(3):
                run = check_output(['retro-contest', 'job', 'submit', '-t', DOCKER_REGISTRY + '/' + agent_tag])
                print('Finished submit %d' %i)

                while not finished():
                    pass



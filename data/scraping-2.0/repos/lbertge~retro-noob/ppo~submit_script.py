import sys
import re
import time
import numpy as np
import json
import docker
import os
import glob
from subprocess import check_output, PIPE, call

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

def main(dockerfile, params_dir, agent_tag):
    DOCKER_REGISTRY = os.environ['DOCKER_REGISTRY']
    client = docker.from_env()
    current_dir = os.getcwd()

    list_of_params = sorted(glob.glob(params_dir + '/*'), key=os.path.getmtime)
    index = 0
    for params_filename in list_of_params:
        params_iter = re.search('/(\d+)$', params_filename)
        index += 1
        if index % 2 == 0:
            continue
        with open(dockerfile, 'w') as f:
            f.write('FROM openai/retro-agent:tensorflow\n')
            f.write('RUN apt-get update && apt-get install -y libgtk2.0-dev && rm -rf /var/lib/apt/lists/*\n')
            f.write('RUN . ~/venv/bin/activate && \\\n')
            f.write('pip install scipy tqdm joblib zmq dill progressbar2 cloudpickle opencv-python && \\\n')
            f.write('pip install --no-deps git+https://github.com/openai/baselines.git\n')

            f.write('ADD ppo2_agent.py ./agent.py\n' )
            f.write('ADD sonic_util.py .\n')
            f.write('ADD %s ./params\n' % params_filename)
            f.write('CMD ["python", "-u", "/root/compo/agent.py"]\n')

        tag = agent_tag + ':' + params_iter.group(1)
        build = check_output(['docker', 'build', '-f', dockerfile, '-t', DOCKER_REGISTRY + '/' + tag, '.'], stdin=PIPE, stderr=PIPE)
        push = check_output(['docker', 'push', DOCKER_REGISTRY + '/' + tag])
        print('Finished build and push')

        run = check_output(['retro-contest', 'job', 'submit', '-t', DOCKER_REGISTRY + '/' + tag])

        while not finished():
            pass

if __name__ == '__main__':
    main('ppo2-scripted.docker', 'params/ppo_v2/final/checkpoints/', 'ppo2_script')

"""
Program Optimisation Environment for Q-Learning
modified from CartPoleEnv.py from OpenAI Gym 
Author: Matthew Tang
Author: Xiao Zheng
Date: 10/4/2022
"""

from File import alter, line_count
import numpy as np
import time
import subprocess
import os
import shutil


class ProgOptEnv:
    """
    Description:
        The source code of a program is provided and will be modified by a set
        of local changes. The goal is to reduce the runtime of the program while
        maintaining its correctness.

    Source:
        NIL

    Observation:
        Type: ndarray()
        Num     Observation               Min                     Max
        0       self.line_pos             0                       99
        1       self.modified_list        Numpy array (stored sorted modification records,
                                            in format (action - 1) * max_line + line

    Actions:
        Type: List(2)
        Num   Action
        0     Skip the change 
        1     Change int into float
        2     Change int into double
        3     Change int into long
        4     Change int into short

    Reward:
        There is no certain goal, so the reward is the proportion of improvement of runtime (from -150% to 100%)
        If there is any error, including runtime error, syntax error and test error, then the reward is set to -1.0.

    Starting State:
        All observations are assigned to 0.

    Reset state:
        Line pos = 0, modified list = None

    Episode Termination:
        There is any syntax error.
        It fails any test cases.
        Episode length is greater than 100. (since the maximum line number is 100)
        The runtime is 200% time more than the baseline.
    """

    def __init__(self):
        self.line_pos = 0
        self.modified_list = np.array([])
        self.goal_actions = None
        # self.max_n_error = 0
        # self.max_n_failed_cases = 0
        self.runtime_threshold = 2.0  # 200% than the baseline
        self.base_file = 'Datatype_B'
        self.base_file_copied = 'Datatype_c'
        self.output_file = 'Datatype_O'
        self.output_file_copied = None
        shutil.copyfile(self.base_file_copied + '.c', self.base_file + '.c')
        self.max_line_pos = line_count(self.base_file + '.c')
        self.compile_state = self.compile_source('gcc ' + self.base_file + '.c -o ' + self.base_file + ' -lm')
        self.runtime_baseline = self.run_and_time('./' + self.base_file + ' > out0')
        self.runtime_proportional = 1

        self.action_space = ['0', '1', '2', '3', '4']
        self.n_actions = len(self.action_space)

        self.state = None

    def reset(self):
        self.line_pos = 0
        self.modified_list = np.array([])
        self.runtime_proportional = 1
        self.base_file = 'Datatype_B'
        self.output_file = 'Datatype_O'
        self.output_file_copied = None
        self.state = np.append(self.line_pos, self.modified_list)
        shutil.copyfile(self.base_file_copied + '.c', self.base_file + '.c')
        return np.array(self.state)

    def compile_source(self, cmd):
        x = subprocess.call(cmd, shell=True)  # just run with all outputs to stdout
        return x  # return value

    def run_and_time(self, cmd):
        t = time.perf_counter()  # start time
        x = subprocess.call(cmd, shell=True)  # just run with all outputs to stdout
        t = time.perf_counter() - t  # end time
        # assert x == 0  # Here x can be not 0, in reality, x represents the process number
        return t

    def digitize(self, p_time):
        if 0 <= p_time < 0.1:
            d_time = 0.05
        elif 0.1 <= p_time < 0.3:
            d_time = 0.2
        elif 0.3 <= p_time < 0.5:
            d_time = 0.4
        elif 0.5 <= p_time < 0.7:
            d_time = 0.6
        elif 0.7 <= p_time < 0.9:
            d_time = 0.8
        elif 0.9 <= p_time < 1.1:
            d_time = 1.0
        elif 1.1 <= p_time < 1.3:
            d_time = 1.2
        elif 1.3 <= p_time < 1.5:
            d_time = 1.4
        elif 1.5 <= p_time < 1.7:
            d_time = 1.6
        elif 1.7 <= p_time < 1.9:
            d_time = 1.8
        elif 1.9 <= p_time:
            d_time = 2.0
        else:
            return None
        return d_time

    def step(self, action):
        # initialisations
        done = None
        reward = None
        info = None

        if action == 0:
            if self.line_pos < self.max_line_pos - 1:
                self.line_pos += 1  # step forward
            else:
                self.line_pos = 0
            done = False
            reward = 0.0
        else:
            # modify the program source code
            # print("modify source code")
            self.output_file_copied = self.output_file
            self.output_file = self.output_file + '_' + str(action) + 'at' + str(self.line_pos + 1)
            alter_result = None
            if action == 1:
                alter_result = alter(self.base_file + '.c', self.output_file + '.c', self.line_pos, 'int', 'float')
            elif action == 2:
                alter_result = alter(self.base_file + '.c', self.output_file + '.c', self.line_pos, 'int', 'double')
            elif action == 3:
                alter_result = alter(self.base_file + '.c', self.output_file + '.c', self.line_pos, 'int', 'long')
            elif action == 4:
                alter_result = alter(self.base_file + '.c', self.output_file + '.c', self.line_pos, 'int', 'short')
            if alter_result == 0:
                # compile -> retrieve syntax error(s)
                n_error = self.compile_source('gcc ' + self.output_file + '.c -o ' + self.output_file + ' -lm')
                # execute -> measure runtime
                if n_error == 0:  # compile successful
                    runtime = self.run_and_time('./' + self.output_file + ' > out1')  # ignore stdout and print
                    # result to out1
                    assert runtime > 0
                    print(f"updated runtime = {runtime}")

                    # check against test cases (if any)
                    with open('out0', 'r', encoding='UTF-8') as x:
                        f = x.readlines()
                    x.close()
                    with open('out1', 'r', encoding='UTF-8') as y:
                        g = y.readlines()
                    y.close()
                    if f == g:
                        # compile successful with no errors
                        self.runtime_proportional = runtime / self.runtime_baseline  # return a normalised value
                        if self.runtime_proportional > self.runtime_threshold:
                            # Time exceeds limit
                            done = True
                            reward = -1.0
                            info = 'terminal'
                        else:
                            # Success!
                            done = False
                            self.base_file = self.output_file  # change files
                            modification_record = (action - 1) * self.max_line_pos + self.line_pos  # (action - 1) *
                            # max_line + line
                            self.modified_list = np.append(self.modified_list, modification_record)
                            self.modified_list.sort()
                            reward = 1 - self.runtime_proportional  # (base_time(1) - new_time) / base_time(1)
                            if np.array_equal(self.modified_list, self.goal_actions):  # win
                                done = True
                                info = 'terminal'
                    else:
                        # compile successful but with syntax error
                        # n_failed = 1
                        print('Test unsuccessful!')
                        done = True
                        reward = -1.0
                        info = 'terminal'
                else:
                    # compile error
                    print('Compile unsuccessful!')
                    done = True
                    reward = -1.0
                    info = 'terminal'
            else:
                # alter unsuccessful (cannot find substitution)
                # print('Alter unsuccessful!')
                done = False
                reward = 0.0
                self.output_file = self.output_file_copied  # Return to previous state

        self.state = np.append(self.line_pos, self.modified_list)
        return self.state, reward, done, info

    def render(self):
        print('render() has not been implemented yet')

    def destroy(self):
        self.reset()
        filename = os.listdir()
        for file in filename:
            if 'out' in file:
                os.remove(file)
            if self.base_file in file:
                os.remove(file)
            if self.output_file in file:
                os.remove(file)

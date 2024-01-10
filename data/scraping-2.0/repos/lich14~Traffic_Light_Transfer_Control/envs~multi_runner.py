"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""

from __future__ import absolute_import, print_function
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

import os
import sys
import torch
import random
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
torch.set_num_threads(8)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    def __init__(self, num_envs):
        self.num_envs = num_envs

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()


def store_observation(id_list, point_lane, conn, points):
    obs_all = []
    for id in range(points ** 2):
        lanes = point_lane[id_list[id]]
        obs = []
        for lane in lanes:
            queue_length = 1 - conn.lane.getLastStepHaltingNumber(lane) / 40
            vehicle_num = 1 - conn.lane.getLastStepVehicleNumber(lane) / 40
            aver_waiting = 1 - conn.lane.getWaitingTime(lane) / (500 * conn.lane.getLastStepVehicleNumber(
                lane)) if conn.lane.getLastStepVehicleNumber(lane) > 0 else 1
            aver_delay = conn.lane.getLastStepMeanSpeed(
                lane) / conn.lane.getMaxSpeed(lane)

            obs += [queue_length, vehicle_num, aver_waiting, aver_delay]
        obs_all.append(obs)
    return obs_all


def sample(conn, idlist, record_vehicle, point_lane, traffic_info=None, points=2, reward_para=0.0):
    periodtime = []
    obs_all = [[] for _ in range(points ** 2)]

    # used to decide when to switch phase
    currentphase = [0 for _ in range(points ** 2)]

    # used to decide when to switch phase
    time_click = [0 for _ in range(points ** 2)]

    ifdone = False
    cur_time = conn.simulation.getTime()
    obs_record = np.zeros([points ** 2, 16])

    for _ in range(90):
        if conn.simulation.getMinExpectedNumber() <= 0:
            ifdone = True
            if obs_all[0]:
                for id in range(points ** 2):
                    obs_all[id] += [0 for _ in range(48 - len(obs_all[id]))]
            else:
                for id in range(points ** 2):
                    obs_all[id] += [0 for _ in range(48)]
            break

        conn.simulationStep()
        cur_time = conn.simulation.getTime()
        for id in range(points ** 2):
            try:
                if currentphase[id] is not conn.trafficlight.getPhase(idlist[id]):
                    time_click[id] = 0
            except:
                return None, None, None, None, None, None

            time_click[id] = time_click[id] + 1

        vehiclein_l = conn.simulation.getDepartedIDList()
        vehicleout_l = conn.simulation.getArrivedIDList()

        if vehiclein_l:
            for i in vehiclein_l:
                record_vehicle[i] = conn.simulation.getTime()

        if vehicleout_l:
            for i in vehicleout_l:
                periodtime.append(
                    conn.simulation.getTime() - record_vehicle[i])
                record_vehicle.pop(i)

        cur_obs_all = np.array(store_observation(
            idlist, point_lane, conn, points))
        obs_record += cur_obs_all

        if conn.simulation.getTime() % 30 == 29:
            for id in range(points ** 2):
                obs_all[id] += (obs_record[id] / 30.).tolist()

        for id in range(points ** 2):
            currentphase[id] = conn.trafficlight.getPhase(idlist[id])
            if traffic_info:
                if time_click[id] >= traffic_info[idlist[id]][currentphase[id] // 2]:
                    conn.trafficlight.setPhase(
                        idlist[id], (currentphase[id] + 1) % 4)

    if periodtime:
        mean_value = np.array(periodtime).mean()
        max_value = np.array(periodtime).max()
        r_value = max_value * reward_para + mean_value * (1 - reward_para)

        reward = 1.0 - r_value / 500
        if reward < -10.0:
            reward = -10.0
        reward = 0.1 * reward
    else:
        reward = 0.0

    return record_vehicle, reward, torch.tensor(obs_all), ifdone, cur_time, periodtime

# action: 0: not change, 1: [+1, -1], 2: [-1, +1], 3: [+5, -5], 4: [-5, +5]


def check_action(input):
    output = np.clip(
        input, -1, 1) if type(input) == np.ndarray or type(input) == np.float64 else input.clamp(-1, 1)
    return output


def step(action, idlist, traffic_info):
    for id, item in enumerate(idlist):

        first_time = check_action(action[id]) + 1  # [0, 2]
        traffic_info[item][0] = int(first_time * 37 + 5)
        traffic_info[item][1] = 84 - traffic_info[item][0]

    return traffic_info


def shareworker(remote, id, points, reward_para=0.0, diff=1.0, sumocfg_str=None):
    np.random.seed(random.randint(0, 100000))
    if sumocfg_str is None:
        sumocfg_str = ''
        for _ in range(points * 2):
            sumocfg_str = sumocfg_str + str(np.random.randint(2))

    sumocfg = f'./{points}{points}network/sumocfg/{sumocfg_str}.sumocfg'
    print('=' * 32)
    print(sumocfg)
    print('=' * 32)
    label = f'sim_{id}'

    time_limit = 80

    # FT:
    # max: 1 point: 2 time: 4978 step: 55
    # max: 2 point: 2 time: 7215 step: 80
    # max: 1 point: 3 time: 5260 step: 58
    # max: 2 point: 3 time: 7581 step: 84

    # max: 1 point: 6 time: 5734
    # TODO: recheck the value of time_limit

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            step_num += 1

            if ifdone:
                remote.send((None, 0, ifdone, cur_time, False,
                            np.array(period_time).mean()))

            else:
                traffic_info = step(data, idlist, traffic_info)
                record_vehicle, reward, obs_all, ifdone, cur_time, cur_period_time = sample(
                    conn, idlist, record_vehicle, point_lane, traffic_info, points, reward_para)
                period_time += cur_period_time

                if period_time == []:
                    period_time_mean = -1
                else:
                    period_time_mean = np.array(period_time).mean()

                if reward is None:
                    remote.send((None, None, None, None, True, None))

                else:
                    traffic_info_ = torch.tensor(
                        [traffic_info[item] for item in traffic_info.keys()])
                    traffic_info_ = (traffic_info_ - 42).float() / 42.

                    obs_all = torch.cat([obs_all, traffic_info_], dim=-1)

                    if step_num >= time_limit:
                        ifdone = True
                        reward = -10

                    remote.send((obs_all, reward, ifdone, cur_time,
                                False, period_time_mean))

        elif cmd == 'restart':

            label = f'sim_{id}'
            restart_label = 0

            for restart_time in range(10):
                try:
                    traci.start(["sumo", "-c", sumocfg], label=label)
                    break
                except Exception as e:
                    print('error 1:', e.__class__.__name__, e)
                    label = f'sim_{id}_{restart_time}'

            if restart_time >= 9:
                restart_label = 1

            remote.send((restart_label, ))

        elif cmd == 'reset':

            period_time = []
            conn = traci.getConnection(label)
            ifdone = False
            step_num = 0
            idlist = conn.trafficlight.getIDList()

            point_lane = {}
            traffic_info = {}

            for item in idlist:
                point_lane[item] = []
                traffic_info[item] = [42, 42]

            lane_ID = conn.lane.getIDList()
            for item in lane_ID:
                if item[-4:-2] in point_lane.keys():
                    point_lane[item[-4:-2]].append(item)

            # here order is down, left, right, up

            record_vehicle = {}
            record_vehicle, reward, obs_all, _, _, _ = sample(
                conn, idlist, record_vehicle, point_lane, traffic_info, points, reward_para)

            if reward is None:
                remote.send((None, True))

            else:
                traffic_info_ = torch.tensor(
                    [traffic_info[item] for item in traffic_info.keys()])
                traffic_info_ = (traffic_info_ - 42).float() / 42.

                try:
                    obs_all = torch.cat([obs_all, traffic_info_], dim=-1)
                    remote.send((obs_all, False))

                except Exception as e:
                    print('error 11:', e.__class__.__name__, e)
                    remote.send((None, True))

        elif cmd == 'close':
            try:
                conn.close()

            except Exception as e:
                print('error 2:', e.__class__.__name__, e)

            remote.close()

            break

        elif cmd == 'get_basic_info':
            # obs_dim. action_dim, n_agents
            remote.send((50, 1, points ** 2))

        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, nenvs, points=2, sumoconfigs=None, reward_para=0.0, diff=1.0):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        if sumoconfigs:
            self.ps = [
                Process(target=shareworker,
                        args=(work_remote, id, points, reward_para, diff, sumoconfig))
                for id, (work_remote, sumoconfig) in enumerate(zip(self.work_remotes, sumoconfigs))
            ]
        else:
            self.ps = [
                Process(target=shareworker,
                        args=(work_remote, id, points, reward_para, diff))
                for id, work_remote in enumerate(self.work_remotes)
            ]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        ShareVecEnv.__init__(self, nenvs)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, reward, ifdone, cur_time, break_check, period_time = zip(*results)

        if any(break_check):
            return None, None, None, None, False, None

        return obs, reward, ifdone, cur_time, True, period_time

    def reset(self):

        restart_time = 0

        while True:
            for remote in self.remotes:
                remote.send(('restart', None))

            results = [remote.recv() for remote in self.remotes]
            start_error = [item[0] for item in results]

            if np.array(start_error).sum() > 0:
                # some error occurs when calling traci.start
                traci.close(False)
                restart_time += 1
                print(f'have restarted {restart_time} times')

                if restart_time > 10:
                    return None, False
            else:
                break

        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]

        obs, break_check = zip(*results)

        if any(break_check):
            return None, False

        return torch.stack(obs, dim=0), True

    def close(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.terminate()
        for p in self.ps:
            p.join()

        try:
            traci.close()
        except Exception as e:
            print('error 3:', e.__class__.__name__, e)

        sys.stdout.flush()
        self.closed = True

    def get_basic_info(self):
        for remote in self.remotes:
            remote.send(('get_basic_info', None))
        results = [remote.recv() for remote in self.remotes]
        obs_dim, n_actions, n_agents = zip(*results)

        return obs_dim[0], n_actions[0], n_agents[0]

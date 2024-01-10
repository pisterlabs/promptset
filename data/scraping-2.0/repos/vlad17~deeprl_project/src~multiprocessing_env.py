"""
Vectorized environment for processing multiple environments at once with
several CPUs. Adapted code from OpenAI's universe package,
which is buggy. See https://github.com/openai/universe/pull/211.
"""

import logging
import multiprocessing
import traceback

import gym

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def _display_name(exception):
    prefix = ''
    # AttributeError has no __module__; RuntimeError has module of
    # exceptions
    if hasattr(exception, '__module__') and \
       exception.__module__ != 'exceptions':
        prefix = exception.__module__ + '.'
    return prefix + type(exception).__name__

def _render_dict(error):
    return {
        'type': _display_name(error),
        'message': error.message,
        'traceback': traceback.format_exc(error)
    }

class _Worker(object):
    def __init__(self, env_m, worker_idx):
        # These are instantiated in the *parent* process
        # currently. Probably will want to change this. The parent
        # does need to obtain the relevant Spaces at some stage, but
        # that's doable.
        self.worker_idx = worker_idx
        self.env_m = env_m
        self.m = len(env_m)
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.joiner = multiprocessing.Process(target=self.run)
        self._clear_state()

        self.start()

        # Parent only!
        self.child_conn.close()

    def _clear_state(self):
        self.mask = [True] * self.m

    # Control methods

    def start(self):
        """Start worker process"""
        self.joiner.start()

    def _parent_recv(self):
        rendered, res = self.parent_conn.recv()
        if rendered is not None:
            raise RuntimeError('[_Worker {}] Error: {} ({})\n\n{}'.format(
                self.worker_idx, rendered['message'],
                rendered['type'], rendered['traceback']))
        return res

    def _child_send(self, msg):
        self.child_conn.send((None, msg))

    def _parent_send(self, msg):
        try:
            self.parent_conn.send(msg)
        except IOError: # the worker is now dead
            try:
                res = self._parent_recv()
            except EOFError:
                _logger.error('[_Worker {}] Child died unexpectedly'
                              .format(self.worker_idx))
            else:
                _logger.error('[_Worker {}] Child returned unexpected result:'
                              ' {}'.format(self.worker_idx, res))

    def close_start(self):
        """Close worker env, notify parent (currently no cleanup)"""
        self._parent_send(('close', None))

    def close_finish(self):
        """Finish up process"""
        self.joiner.join()

    def reset_start(self):
        """Reset worker env"""
        self._parent_send(('reset', None))

    def reset_finish(self):
        """Notify parent of worker env reset"""
        return self._parent_recv()

    def step_start(self, action_m):
        """action_m: the batch of actions for this worker"""
        self._parent_send(('step', action_m))

    def step_finish(self):
        """notify parent of step finish"""
        return self._parent_recv()

    def mask_start(self, i):
        """Mask an env from being stepped"""
        self._parent_send(('mask', i))

    def seed_start(self, seed_m):
        """Seed envs"""
        self._parent_send(('seed', seed_m))

    def render_start(self, mode, close):
        """render envs"""
        self._parent_send(('render', (mode, close)))

    def render_finish(self):
        """notify parent render finished"""
        return self._parent_recv()

    def run(self):
        """run receive loop on remote worker process"""
        try:
            self.do_run()
        except Exception as e: # pylint: disable=broad-except
            rendered = _render_dict(e)
            self.child_conn.send((rendered, None))
            return

    def do_run(self):
        """receive loop called in separate process"""
        # Child only!
        self.parent_conn.close()

        while True:
            method, body = self.child_conn.recv()
            if method == 'close':
                _logger.info('Closing envs')
                return
            elif method == 'reset':
                self._clear_state()
                observation_m = [env.reset() for env in self.env_m]
                self._child_send(observation_m)
            elif method == 'step':
                action_m = body
                observation_m, reward_m, done_m, info = self.step_m(action_m)
                self._child_send((observation_m, reward_m, done_m, info))
            elif method == 'mask':
                i = body
                self.mask[i] = False
            elif method == 'seed':
                seeds = body
                for env, seed in zip(self.env_m, seeds):
                    env.seed(seed)
            elif method == 'render':
                mode, close = body
                if mode == 'human':
                    self.env_m[0].render(mode=mode, close=close)
                    result = [None]
                else:
                    result = [env.render(mode=mode, close=close)
                              for env in self.env_m]
                self._child_send(result)
            else:
                raise RuntimeError('Bad method: {}'.format(method))

    def step_m(self, action_m):
        """Step all envs in the worker"""
        observation_m = []
        reward_m = []
        done_m = []
        info = {'m': []}

        for env, enabled, action in zip(self.env_m, self.mask, action_m):
            if enabled:
                observation, reward, done, info_i = env.step(action)
            else:
                observation = None
                reward = 0
                done = False
                info_i = {}
            observation_m.append(observation)
            reward_m.append(reward)
            done_m.append(done)
            info['m'].append(info_i)
        return observation_m, reward_m, done_m, info


def _step_n(worker_n, action_n):
    accumulated = 0
    for worker in worker_n:
        action_m = action_n[accumulated:accumulated+worker.m]
        worker.step_start(action_m)
        accumulated += worker.m

    observation_n = []
    reward_n = []
    done_n = []
    info = {'n': []}

    for worker in worker_n:
        observation_m, reward_m, done_m, info_i = worker.step_finish()
        observation_n += observation_m
        reward_n += reward_m
        done_n += done_m
        info['n'] += info_i['m']
    return observation_n, reward_n, done_n, info


def _reset_n(worker_n):
    for worker in worker_n:
        worker.reset_start()

    observation_n = []
    for worker in worker_n:
        observation_n += worker.reset_finish()

    return observation_n


def _seed_n(worker_n, seeds):
    accumulated = 0
    for worker in worker_n:
        seed_m = seeds[accumulated:accumulated+worker.m]
        worker.seed_start(seed_m)
        accumulated += worker.m

def _mask(worker_n, i):
    accumulated = 0
    for _, worker in enumerate(worker_n):
        if accumulated + worker.m <= i:
            accumulated += worker.m
        else:
            worker.mask_start(i - accumulated)
            return

def _render_n(worker_n, mode, close):
    if mode == 'human':
        # Only render 1 worker
        worker_n = worker_n[0:]

    for worker in worker_n:
        worker.render_start(mode, close)
    res = []
    for worker in worker_n:
        res += worker.render_finish()
    if mode != 'human':
        return res
    return None

def _close_n(worker_n):
    if worker_n is None:
        return

    for worker in worker_n:
        try:
            worker.close_start()
        except: # pylint: disable=bare-except
            pass

class MultiprocessingEnv(gym.Env):
    """
    Vectorized environment which uses multiple CPU processors to generate
    several rollouts at once.
    """

    def __init__(self, envs):
        assert envs, len(envs)
        self.envs = envs
        self.worker_n = None

        env = envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.n = len(envs)

        pool_size = min(len(self.envs), multiprocessing.cpu_count() - 1)
        pool_size = max(1, pool_size)

        self.worker_n = []
        m = int((self.n + pool_size - 1) / pool_size)
        for i in range(0, self.n, m):
            envs = self.envs[i:i+m]
            self.worker_n.append(_Worker(envs, i))

    def _seed(self, seed=None):
        _seed_n(self.worker_n, seed)
        return [[seed_i] for seed_i in seed]

    def _reset(self):
        return _reset_n(self.worker_n)

    def _step(self, action):
        return _step_n(self.worker_n, action)

    def _render(self, mode='human', close=False):
        return _render_n(self.worker_n, mode=mode, close=close)

    def mask(self, i):
        """Mask an environment from being stepped until next reset."""
        _mask(self.worker_n, i)

    def _close(self):
        _close_n(self.worker_n)

from collections import defaultdict
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import numpy as np

from e_maml_tf.ge_policies import MlpPolicy


def path_gen_fn(env: SubprocVecEnv, policy: MlpPolicy, start_reset=False, soft=True):
    """
    Generator function for the path data. This one outputs the log-likelihood, value and baseline.

    Usage:

    |   s = path_gen_fn(...)
    |   timesteps = 100
    |   paths = s.send(timesteps)
    |
    |   assert "acs" in paths
    |   assert "obs" in paths

    :param env: A parallel env, with first index being the batch
    :param policy: has the signature `act`, returns batched actions for each observation (in the batch)
    :param gamma: the gamma parameter for the GAE
    :param lam: the lambda parameter for the GAE
    :param start_reset: boolean flag for resting on each generator start.
    :param soft:
    :param _render:
    :return: dimension is Size(timesteps, n_envs, feature_size)
    """
    # todo: use a default dict for these data collection. Much cleaner.

    timesteps = yield
    obs, dones = env.reset(), [False] * env.num_envs
    paths = defaultdict(list)
    while True:
        paths.clear()
        # do NOT use this if environment is parallel env.
        if start_reset:  # note: mostly useless.
            obs, dones = env.reset(), [False] * env.num_envs
        for _ in range(timesteps):
            paths['obs'].append(obs.copy())
            if policy.vf is None:
                actions, neglogpacs = policy.act(obs, soft)
            else:
                actions, values, neglogpacs = policy.act(obs, soft)
                paths['values'].append(values)
            paths['acs'].append(actions.copy())
            paths['neglogpacs'].append(neglogpacs)
            obs, rewards, dones, info = env.step(actions)

            paths['rewards'].append(rewards)
            paths['dones'].append(dones)

            # In multiworld, `info` contains the entire observation. Processing these
            # will take way too much time. So we don't do that.
            _suc = [_['success'] for _ in info if 'success' in _]
            if _suc:
                paths['info.successes'].append(_suc)
            _dist = [_['dist'] for _ in info if 'dist' in _]
            if _dist:
                paths['info.dists'].append(_dist)


        # The TimeLimit env wrapper "dones" the env when time limit
        # has been reached. This is technically not correct.
        # if has vf and not done. Discounted infinite horizon.
        done_mask = 1 - dones
        if policy.vf is not None and done_mask.all():  # bootstrap from the (k + 1)th value
            paths['last_values'] = policy.value(obs) * done_mask

        timesteps = yield {k: np.array(v) for k, v in paths.items()}
        # now, this is missing bunch of stuff, return for example.


def paths_reshape(paths, horizon):
    """
    reshapes the trajectories in the path. Used to split paths data with multiple
    rollouts in a single env into k independent rollout vectors. This is needed
    for fitting the linear feature baseline.

    | n -> timesteps, k -> rollouts, c -> features.
    
    :param paths: dict('acs', 'obs', ...)
    :param horizon: int, the horizon we want to chop the paths dict into
    :return:
    """
    _ = paths.copy()
    for key, d in _.items():
        if not isinstance(d, np.ndarray) or len(d.shape) < 2:
            continue  # I prefer explicitness, but this requires less maintenance
        n, k, *c = d.shape  # *c accommodate rank-2 rewards/returns tensors
        _[key] = d.swapaxes(0, 1) \
            .reshape(n * k // horizon, horizon, *c) \
            .swapaxes(0, 1)
    return _


def mc(paths, gamma=None):
    rewards = paths['rewards']
    dones = paths['dones']  # not used
    returns = np.zeros_like(rewards)
    value_so_far = paths['last_values'] if 'last_values' in paths else np.zeros_like(rewards[-1])
    for step in range(len(returns) - 1, -1, -1):
        done_mask = 1 - dones[step]
        value_so_far = rewards[step] + gamma * value_so_far * done_mask
        returns[step] = value_so_far
    return returns


from e_maml_tf.value_baselines.linear_feature_baseline import LinearFeatureBaseline


def value_baseline(paths, m: LinearFeatureBaseline = None):
    m = m or LinearFeatureBaseline()
    m.fit(paths['obs'], paths['rewards'], paths['returns'])
    return m.predict(paths['obs'], paths['rewards'])


def gae(paths, gamma, lam):
    assert 'values' in paths, 'paths data need to contain value estimates.'
    gl = gamma * lam
    rewards = paths['rewards']
    dones = paths['dones']
    values = paths['values']
    last_values = paths['last_values'] if 'last_values' in paths else np.zeros_like(rewards[-1])
    gae = np.zeros_like(rewards)
    last_gae = 0
    l = len(rewards)
    for step in range(l - 1, -1, -1):
        done_mask = 1 - dones[step]
        delta = rewards[step] + gamma * (last_values if step == l - 1 else values[step]) * done_mask - values[step]
        last_gae = delta + gl * last_gae * done_mask
        gae[step] = last_gae
    return gae


linear_baseline_model = LinearFeatureBaseline()


def paths_process(paths, baseline, horizon, gamma=None, use_gae=None, lam=None, **_):
    """
    Master RL sample Processor, with GAE configurations and value baseline.

    :param paths:
    :param baseline:
    :param use_gae:
    :param gamma:
    :param lam:
    :return:
    """
    _ = paths.copy()
    _['returns'] = mc(_, gamma=gamma)
    # fixit: this is wrong. Need to fix
    if horizon:
        _ = paths_reshape(_, horizon)  # Need to reshape by rollout for the fitted linearFeatureBaseline.
    if baseline == 'linear':
        assert 'values' not in _, '_ should not contain value estimates when ' \
                                  'using the linear feature baseline. LFB Overwrites original estimate.'
        # todo: use a single baseline model instance to save on speed.
        _['values'] = value_baseline(_, m=linear_baseline_model)
    if use_gae:
        _['advs'] = gae(_, gamma=gamma, lam=lam)
    return _


if __name__ == "__main__":
    import gym
    from e_maml_tf.custom_vendor import IS_PATCHED

    make_env = lambda: gym.make('PointMass-v0')
    envs = SubprocVecEnv([make_env for i in range(1)])
    print(envs)
    envs.reset()
    policy_stub = lambda: None
    policy_stub.vf = None
    policy_stub.value = None
    # policy_stub.act = lambda obs, _: [np.random.rand(1, 2), np.zeros(1)]
    policy_stub.act = lambda obs, _: [obs[:, -2:] - 0.5 * obs[:, 2:4], np.zeros(1)]
    path_gen = path_gen_fn(envs, policy_stub, start_reset=True)
    next(path_gen)
    timesteps = 1000
    paths = path_gen.send(timesteps)

    # Usage Example: Using GAE
    gamma, lam = 0.995, 0.99
    paths['returns'] = mc(paths, gamma=gamma)
    paths = paths_reshape(paths, 50)  # Need to reshape by rollout for the fitted linearFeatureBaseline.
    if "values" not in paths:  # use linear baseline when critic is not available.
        paths['values'] = value_baseline(paths)
    phi = gae(paths, gamma=gamma, lam=lam)

    # Usage Example: Not Using GAE
    # gamma, lam = 0.995, 0.99
    # paths['returns'] = mc(paths, gamma=gamma)
    # phi = paths['returns']

    # plot the results
    import matplotlib.pyplot as plt

    plt.plot(paths['rewards'], color='green')
    plt.plot(paths['returns'], color='red')
    plt.plot(paths['values'], color='gray')
    plt.plot(phi, color='blue')
    plt.show()

    exit()


def _deprecated_ppo2_gae():
    # from OpenAI.baselines.ppo2
    # compute returns from path.
    # 0. compute rewards
    # 1. compute adv (GAE)
    # 2. compute regular adv (no GAE)
    """
    rewards = r + \gamma * V(s_{t + 1})
    """
    advs = np.zeros_like(paths['rewards'])

    # discount/bootstrap off value fn
    _advs = np.zeros_like(paths['rewards'])
    last_gae_lam = 0
    n_rollouts = len(_obs)
    for t in reversed(range(n_rollouts)):
        if t == n_rollouts - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - _dones[t + 1]
            next_values = _values[t + 1]
        delta = _rewards[t] + gamma * next_values * next_non_terminal - _values[t]
        _advs[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    _returns = _advs + _values

    # return dimension is Size(timesteps, n_envs, feature_size)
    timesteps = yield dict(obs=_obs, acs=_actions, rewards=_rewards, dones=_dones,
                           returns=_returns,
                           values=_values, neglogpacs=_neglogpacs, ep_info=ep_info)


def _deprecated_gae_old(paths, gamma, lam):
    """
    Compute advantage with GAE(lambda)
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(paths["new"], 0)
    vpred = np.append(paths["vpred"], paths["nextvpred"])
    T = len(paths["rew"])
    paths["adv"] = gaelam = np.empty(T, 'float32')
    rew = paths["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    paths["tdlamret"] = paths["adv"] + paths["vpred"]

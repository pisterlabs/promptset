from core import GymEnv
from atari import get_atari_env
from doom import get_doom_env

def get_env(env_id):
    '''env_id is a string of the format `prefix.env_name`'''
    # parse
    parts = env_id.split('.')
    rest = '.'.join(parts[1:])

    # envs from OpenAI gym with no wrappers
    if parts[0] == 'gym':
        return GymEnv(parts[1])

    # Atari games
    if parts[0] == 'atari':
        return get_atari_env(rest)

    # Doom envs
    if parts[0] == 'doom':
        return get_doom_env(rest)


if __name__ == '__main__':
    import sys
    from core import test_env

    env = get_env(sys.argv[1])
    test_env(env)

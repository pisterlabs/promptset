from __future__ import annotations, unicode_literals
from datetime import datetime
from functools import lru_cache
import glob
import importlib
import importlib.util
import logging
import os
import pathlib
import pkgutil
import sys
from typing import Any, Union

import click
import questionary

from gupb import controller
from gupb import runner
from openaigym_env import GameEnv
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy

# noinspection PyUnresolvedReferences
@lru_cache()
def possible_controllers() -> list[controller.Controller]:
    controllers = []
    pkg_path = os.path.dirname(controller.__file__)
    names = [name for _, name, _ in pkgutil.iter_modules(path=[pkg_path], prefix=f"{controller.__name__}.")]
    for name in names:
        module = importlib.import_module(name)
        controllers.extend(module.POTENTIAL_CONTROLLERS)
    return controllers


# noinspection PyUnresolvedReferences
def load_initial_config(config_path: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config_module
    spec.loader.exec_module(config_module)
    return config_module.CONFIGURATION


# noinspection PyUnresolvedReferences
def possible_arenas() -> set[str]:
    paths = glob.glob("resources/arenas/*.gupb")
    return set(pathlib.Path(path).stem for path in paths)


def configuration_inquiry(initial_config: dict[str, Any]) -> dict[str, Any]:
    def when_show_sight(current_answers: dict[str, Any]) -> bool:
        chosen_controllers.extend([
            {
                'name': possible_controller.name,
                'value': possible_controller,
            }
            for possible_controller in current_answers['controllers']
        ])
        default_controller = [c for c in chosen_controllers if c['value'] == initial_config['show_sight']]
        other_controllers = [c for c in chosen_controllers if c['value'] != initial_config['show_sight']]
        chosen_controllers.clear()
        chosen_controllers.extend(default_controller)
        chosen_controllers.extend(other_controllers)
        return current_answers['visualise']

    def validate_runs_no(runs_no: str) -> Union[bool, str]:
        try:
            int(runs_no)
            return True
        except ValueError:
            return "The number of games should be a valid integer!"

    chosen_controllers = [
        {
            'name': 'None',
            'value': None,
        },
    ]
    questions = [
        {
            'type': 'checkbox',
            'name': 'arenas',
            'message': 'Which arenas should be used in this run?',
            'choices': [
                {
                    'name': possible_arena,
                    'checked': possible_arena in initial_config['arenas']
                }
                for possible_arena in possible_arenas()
            ],
        },
        {
            'type': 'checkbox',
            'name': 'controllers',
            'message': 'Which controllers should participate in this run?',
            'choices': [
                {
                    'name': possible_controller.name,
                    'value': possible_controller,
                    'checked': possible_controller in initial_config['controllers'],
                }
                for possible_controller in possible_controllers()
            ],
        },
        {
            'type': 'confirm',
            'name': 'visualise',
            'message': 'Show the live game visualisation?',
            'default': initial_config['visualise'],
        },
        {
            'type': 'select',
            'name': 'show_sight',
            'message': 'Which controller should have its sight visualised?',
            'when': when_show_sight,
            'choices': chosen_controllers,
            'filter': lambda result: None if result == 'None' else result,
            'default': initial_config['show_sight'],
        },
        {
            'type': 'input',
            'name': 'runs_no',
            'message': 'How many games should be played?',
            'validate': validate_runs_no,
            'filter': int,
            'default': str(initial_config['runs_no']),
        },
    ]
    answers = questionary.prompt(questions)
    return answers


def configure_logging(log_directory: str) -> None:
    logging_dir_path = pathlib.Path(log_directory)
    logging_dir_path.mkdir(parents=True, exist_ok=True)
    logging_dir_path.chmod(0o777)
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    verbose_logger = logging.getLogger('verbose')
    verbose_logger.propagate = False
    verbose_file_path = logging_dir_path / f'gupb__{time}.log'
    verbose_file_handler = logging.FileHandler(verbose_file_path.as_posix())
    verbose_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s'
    )
    verbose_file_handler.setFormatter(verbose_formatter)
    verbose_logger.addHandler(verbose_file_handler)
    verbose_logger.setLevel(logging.DEBUG)

    json_logger = logging.getLogger('json')
    json_logger.propagate = False
    json_file_path = logging_dir_path / f'gupb__{time}.json'
    json_file_handler = logging.FileHandler(json_file_path.as_posix())
    json_formatter = logging.Formatter(
        '{"time_stamp": "%(asctime)s",'
        ' "severity": "%(levelname)s",'
        ' "line": "%(module)s.%(funcName)s:%(lineno)d",'
        ' "type": "%(event_type)s",'
        ' "value": %(message)s}'
    )
    json_file_handler.setFormatter(json_formatter)
    json_logger.addHandler(json_file_handler)
    json_logger.setLevel(logging.DEBUG)


@click.command()
@click.option('-c', '--config_path', default='gupb/default_config.py',
              type=click.Path(exists=True), help="The path to run configuration file.")
@click.option('-i', '--inquiry',
              is_flag=True, help="Whether to configure the runner interactively on start.")
@click.option('-l', '--log_directory', default='results',
              type=click.Path(exists=False), help="The path to log storage directory.")

def main_run(config_path: str, inquiry: bool, log_directory: str) -> None:
    configure_logging(log_directory)
    current_config = load_initial_config(config_path)
    current_config = configuration_inquiry(current_config) if inquiry else current_config
    game_runner = runner.Runner(current_config)
    game_runner.run()
    game_runner.print_scores()



def main_learn():
    env = GameEnv()
    print(env)

    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile(Adam(lr=1e-3), metrics=['mae'])
    # sarsa.load_weights('sarsa_{}_weights.h5f'.format("ihavenoideawhatimdoing"))
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    sarsa.fit(env, nb_steps=5000, visualize=False, verbose=2)

    sarsa.save_weights('sarsa_{}_weights.h5f'.format("ihavenoideawhatimdoing"), overwrite=True)

    # sarsa.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main_run()

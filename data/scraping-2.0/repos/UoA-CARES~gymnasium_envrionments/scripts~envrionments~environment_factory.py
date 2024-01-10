import logging

from envrionments.dmcs.dmcs_environment import DMCSEnvironment
from envrionments.gym_environment import GymEnvironment
from envrionments.image_wrapper import ImageWrapper
from envrionments.openai.openai_environment import OpenAIEnvrionment
from envrionments.pyboy.mario.mario_environment import MarioEnvironment
from envrionments.pyboy.pokemon.pokemon_environment import PokemonEnvironment
from util.configurations import GymEnvironmentConfig


def create_pyboy_environment(config: GymEnvironmentConfig) -> GymEnvironment:
    # TODO extend to other pyboy games...maybe another repo?
    if config.task == "pokemon":
        env = PokemonEnvironment(config)
    elif config.task == "mario":
        env = MarioEnvironment(config)
    else:
        raise ValueError(f"Unkown pyboy environment: {config.task}")
    return env


class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: GymEnvironmentConfig) -> GymEnvironment:
        logging.info(f"Training Environment: {config.gym}")
        if config.gym == "dmcs":
            env = DMCSEnvironment(config)
        elif config.gym == "openai":
            env = OpenAIEnvrionment(config)
        elif config.gym == "pyboy":
            env = create_pyboy_environment(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym}")
        return ImageWrapper(env) if bool(config.image_observation) else env

# # FAPS PLMAgents
# ## FAPS PLM ML-Agent Learning
import logging
import os
import random
import re
from enum import Enum

import gym
import gym_eflex_agent
import keras
import keras.backend as backend
import numpy as np
import tensorflow as tf
import yaml

from OpenAIGym.exception import FAPSPLMEnvironmentException


class SPACE_TYPE(Enum):
    Discrete = 1
    Continous = 0


class TrainerController(object):
    def __init__(self, use_gpu, brain_names, environment_names, render, save_freq, load, train, keep_checkpoints, seed,
                 trainer_config_path):
        """
        :param brain_names: Names of the brain to train
        :param save_freq: Frequency at which to save model
        :param load: Whether to load the model or randomly initialize
        :param train: Whether to train model, or only run inference
        :param environment_names: Environment to user
        :param keep_checkpoints: How many model checkpoints to keep
        :param seed: Random seed used for training
        :param trainer_config_path: Fully qualified path to location of trainer configuration file
        """

        self.use_gpu = use_gpu
        self.trainer_config_path = trainer_config_path
        self.logger = logging.getLogger("FAPSPLMAgents")
        self.environment_names = environment_names
        self.save_freq = save_freq
        self.load_model = load
        self.train_model = train
        self.render = render
        self.keep_checkpoints = keep_checkpoints
        self.trainers = {}

        # Generate seed
        if seed == -1:
            seed = np.random.randint(0, 999999)

        # Set the seed
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        if backend.backend() == 'tensorflow':
            tf.set_random_seed(self.seed)
        else:
            np.random.seed(seed)

        # parse and format the brain names
        self.brain_names = []
        for b in brain_names:
            self.brain_names.append(re.sub('[^0-9a-zA-Z]+', '_', b))

        # Reset the environment and get all parameters from the simulation
        self.envs = {}
        for e in environment_names:
            self.envs[e] = gym.make(e)
            self.envs[e].seed(self.seed)

        self.rewards = {}
        self.dones = {}
        self.infos = {}
        self.observations = {}

        self.model_paths = {}
        for b in self.brain_names:
            self.model_paths[b] = 'models/%s' % b

    def _get_progress(self, brain_name, step_progress, reward_progress):
        """
        Compute and increment the progess of a specified trainer.
        :param brain_name: Name of the brain to train
        :param step_progress: last step
        :param reward_progress: last cummulated reward
        """
        step_progress += self.trainers[brain_name].get_step / self.trainers[brain_name].get_max_steps
        reward_progress += self.trainers[brain_name].get_last_reward
        return step_progress, reward_progress

    def _save_model(self, _trainer, _model_path):
        """
        Saves current model to checkpoint folder.
        """
        _trainer.save_model(_model_path)
        print("INFO: Model saved.")

    @staticmethod
    def _import_module(module_name, class_name):
        """Constructor"""

        macro_module = __import__(module_name)
        module = getattr(macro_module, 'FAPSPLMAgents')
        my_class = getattr(module, class_name)
        my_class = getattr(my_class, class_name)
        return my_class

    def _initialize_trainer(self, _brain_name, _trainer_config):

        _trainer_parameters = _trainer_config['default'].copy()
        _graph_scope = re.sub('[^0-9a-zA-Z]+', '_', _brain_name)
        _trainer_parameters['graph_scope'] = _graph_scope
        _trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
            basedir='summaries',
            name=str(_graph_scope))
        if _brain_name in _trainer_config:
            _brain_key = _brain_name
            while not isinstance(_trainer_config[_brain_key], dict):
                _brain_key = _trainer_config[_brain_key]
            for k in _trainer_config[_brain_key]:
                _trainer_parameters[k] = _trainer_config[_brain_key][k]
        _trainer_parameters = _trainer_parameters.copy()

        # Instantiate the trainer
        # import the module
        module_spec = self._import_module("OpenAIGym.FAPSPLMAgents." + _trainer_parameters['trainer'],
                                          _trainer_parameters['trainer'])

        if module_spec is None:
            raise FAPSPLMEnvironmentException("The trainer config contains an unknown trainer type for brain {}"
                                              .format(_brain_name))
        else:
            return module_spec(self.envs, _brain_name, _trainer_parameters, self.train_model, self.seed)

    @staticmethod
    def _load_config(_trainer_config_path):
        try:
            with open(_trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file, Loader=yaml.SafeLoader)
                return trainer_config
        except IOError:
            raise FAPSPLMEnvironmentException("""Parameter file could not be found here {}.
                                            Will use default Hyper parameters"""
                                              .format(_trainer_config_path))
        except UnicodeDecodeError:
            raise FAPSPLMEnvironmentException("There was an error decoding Trainer Config from this path : {}"
                                              .format(_trainer_config_path))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise FAPSPLMEnvironmentException("The folder {} containing the generated model could not be accessed. "
                                              "Please make sure the permissions are set correctly.".format(model_path))

    def start_learning(self):

        # configure tensor flow to use 8 cores
        if self.use_gpu:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"GPU": 1},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")
        else:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"CPU": 8},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")

        # Load the configurations
        trainer_config = self._load_config(self.trainer_config_path)

        # Create the model path
        for b in self.brain_names:
            self._create_model_path(self.model_paths[b])

        # Choose and instantiate the trainers
        for b in self.brain_names:
            self.trainers[b] = self._initialize_trainer(b, trainer_config)

        print("\n##################################################################################################")
        print("Starting Training...")
        print("Backend : {}".format(backend.backend()))
        print("Use cpu: {}".format(self.use_gpu))
        iterator = 0
        for k, t in self.trainers.items():
            print("Trainer({}): {}".format(iterator, t.__str__()))
            iterator = iterator + 1
        print("##################################################################################################")

        # Initialize the trainer
        for k, t in self.trainers.items():
            t.initialize()

        # Instantiate model parameters
        if self.load_model:
            print("\nINFO: Loading models ...")
            for k, t in self.trainers.items():
                t.load_model_and_restore(self.model_paths[k])

        global_step = 0  # This is only for saving the model
        cumulated_reward = 0.0
        cumulated_rewards = {}
        # Reset the environments
        for e, env in self.envs.items():
            self.observations[e] = env.reset()
            self.dones[e] = False
            self.rewards[e] = 0.0
            self.infos[e] = None
            cumulated_rewards[e] = None

        # Write Tensor board settings
        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensor_board_text('Hyperparameters', trainer.parameters)
        try:
            while any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()]) or not self.train_model:
                for brain_name, trainer in self.trainers.items():
                    action_map = {}
                    new_observations = {}

                    for e, env in self.envs.items():
                        # reset the environment and the trainers if env is done
                        if self.dones[e]:
                            self.observations[e] = env.reset()
                            trainer.end_episode()

                        # Decide and take an action
                        action_map[brain_name] = trainer.take_action(self.observations[e], env)
                        new_observations[e], self.rewards[e], self.dones[e], self.infos[e] = \
                            env.step(action_map[brain_name])

                        # Process experience and generate statistics
                        trainer.add_experiences(self.observations[e], action_map[brain_name], new_observations[e],
                                                self.rewards[e], self.dones[e], self.infos[e])
                        trainer.process_experiences(self.observations[e], action_map[brain_name], new_observations[e])
                        if isinstance(self.rewards[e], (list,)):
                            mean = np.mean(np.array(self.rewards[e]))
                            cumulated_reward += mean
                            if cumulated_rewards[e] is None:
                                cumulated_rewards[e] = np.array(self.rewards[e])
                            else:
                                cumulated_rewards[e] += np.array(self.rewards[e])
                        else:
                            cumulated_reward += self.rewards[e]
                            cumulated_rewards[e] += np.array(self.rewards[e])

                        if trainer.is_ready_update() and trainer.get_step <= trainer.get_max_steps:
                            if self.train_model:
                                # Perform gradient descent with experience buffer
                                trainer.update_model()

                                # Write training statistics.
                                trainer.write_summary()

                            if self.render:
                                # Write to tensorborad
                                trainer.write_tensorboard_value('cul_reward', cumulated_reward)

                                all_rewards = cumulated_rewards[e]
                                for i in range(len(all_rewards)):
                                     trainer.write_tensorboard_value('cul_reward_agent_{:d}'.format(i), all_rewards[i])

                            # Reset the statistics values
                            cumulated_reward = 0
                            cumulated_rewards[e] = None

                        # Increment the step counter
                        trainer.increment_step()

                        if self.train_model and trainer.get_step <= trainer.get_max_steps:
                            trainer.update_last_reward(self.rewards[e])

                        # Render the environment
                        if self.render:
                            env.render()
                            # print("CUL. REWARD: {}".format(cumulated_reward))

                    self.observations = new_observations

                # Update Global Step
                if self.train_model :
                    global_step += 1

                # Save the models by the save frequency
                if global_step % self.save_freq == 0 and global_step != 0 and self.train_model:
                    # Save the models
                    for brain_name, trainer in self.trainers.items():
                        self._save_model(trainer, self.model_paths[brain_name])

            # Final save  model
            if global_step != 0 and self.train_model:
                for brain_name, trainer in self.trainers.items():
                    self._save_model(trainer, self.model_paths[brain_name])
        except KeyboardInterrupt:
            if self.train_model:
                self.logger.info("Learning was interrupted. Please wait while the graph is generated.")
                for brain_name, trainer in self.trainers.items():
                    self._save_model(trainer, self.model_paths[brain_name])
            pass

        # Clear the trainer
        for k, t in self.trainers.items():
            t.clear()

        # clear the backend
        backend.clear_session()

        for e, env in self.envs.items():
            env.close()  # If needed save some parameters

        print("\n##################################################################################################")
        print("Training ended. ")
        print("##################################################################################################")

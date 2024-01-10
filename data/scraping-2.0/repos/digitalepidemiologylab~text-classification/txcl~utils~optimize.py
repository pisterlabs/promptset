import os
import json
from .config_reader import ConfigReader
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import time
import pickle
from copy import copy
import logging
from .misc import get_json_hash, JSONEncoder
import uuid
import shutil
import functools

class Optimize():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_reader = ConfigReader()
        self.optimize_space = {}
        self.logger = logging.getLogger(__name__)

    def init(self):
        # read config
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError('Config file under {} does not exist'.format(self.config_path))
        with open(self.config_path, 'r') as cf:
            config = json.load(cf)
        config = self.config_reader.parse_from_dict(config)
        self.verify_config(config)
        self.original_config = config
        self.config = config.runs[0]
        self.optimize_space = self.parse_optimize_space(self.config.optimize_space)
        self.algo = tpe.suggest
        self.max_evals = self.config.get('optimize_max_eval', 10)
        self.keep_models = self.config.get('optimize_keep_models', False)
        self.output_path = self.config.output_path
        self.metric = self.config.get('optimize_metric', 'f1_macro')
    
    def parse_optimize_space(self, optimize_space):
        _optimize_space = {}
        for opt in optimize_space:
            if opt.type == 'choice':
                _optimize_space[opt.param] = hp.choice(opt.param, eval(str(opt['values'])))
            elif opt.type == 'uniform':
                _optimize_space[opt.param] = hp.uniform(opt.param, *eval(str(opt['values'])))
            elif opt.type == 'normal':
                _optimize_space[opt.param] = hp.normal(opt.param, *eval(str(opt['values'])))
            else:
                raise ValueError('Unknown optimize space {}'.format(opt.type))
        return _optimize_space

    def run(self):
        trials = Trials()
        fmin_objective = functools.partial(self.objective, metric=self.metric)
        best = fmin(fmin_objective, space=self.optimize_space, algo=self.algo, max_evals=self.max_evals, trials=trials)
        best = space_eval(self.optimize_space, best)
        best_score = -1*min(trials.losses())
        self.logger.info('Best {} score: {} with params: {}'.format(self.metric, best_score, best))

    def objective(self, params, metric='f1_macro'):
        self.logger.info('Params: {}'.format(params))
        unique_id = uuid.uuid4().hex
        config = self.update_config(params, unique_id)
        model = self.get_model(config.model)
        model.train(config)
        result = model.test(config)
        self.dump_results(unique_id, result, params, model.model_state)
        if not config.optimize_keep_models:
            shutil.rmtree(config.output_path)
        self.logger.info('{}: {}'.format(metric, result[metric]))
        return {
                'loss': -result[metric],
                'status': STATUS_OK
                }

    def dump_results(self, unique_id, *args):
        d = {}
        for arg in args:
            d = {**d, **arg}
        test_output = os.path.join(self.output_path, 'optimize_results_{}.json'.format(unique_id))
        with open(test_output, 'w') as f:
            json.dump(d, f, cls=JSONEncoder, indent=4)

    def update_config(self, params, unique_id):
        config = copy(self.config)
        for k, v in params.items():
            config[k] = v
        # create new output dir
        config.output_path = os.path.join(config.output_path, unique_id)
        os.makedirs(config.output_path)
        return config

    def get_model(self, model_name):
        """Dynamically import model module and return model instance"""
        if model_name == 'fasttext':
            from ..models.fasttext import FastText
            return FastText()
        if model_name == 'fasttext_pretrain':
            from ..models.fasttext_pretrain import FastTextPretrain
            return FastTextPretrain()
        elif model_name == 'bag_of_words':
            from ..models.bag_of_words import BagOfWordsModel
            return BagOfWordsModel()
        elif model_name == 'bert':
            from ..models.bertmodel import BERTModel
            return BERTModel()
        elif model_name == 'openai_gpt2':
            from ..models.openai_gpt2 import OpenAIGPT2
            return OpenAIGPT2()
        elif model_name == 'dummy':
            from ..models.dummy_models import DummyModel
            return DummyModel()
        elif model_name == 'random':
            from ..models.dummy_models import RandomModel
            return RandomModel()
        elif model_name == 'weighted_random':
            from ..models.weighted_random import WeightedRandomModel
            return WeightedRandomModel()
        else:
            raise NotImplementedError('Model `{}` is unknown'.format(model_name))

    def verify_config(self, config):
        if len(set([run_config.train_data for run_config in config.runs])) != 1:
            raise ValueError('Cannot accept different training data sources when running learning curve for multiple models.')
        if len(set([run_config.test_data for run_config in config.runs])) != 1:
            raise ValueError('Cannot accept different test data sources when running learning curve for multiple models.')
        run_config = config.runs[0]
        if 'optimize_space' not in run_config:
            raise ValueError("Config doesn't contain the key 'optimize_sace'.")
        if not isinstance(run_config.optimize_space, list):
            raise ValueError("Key 'optimize_sace' should be of type list.")
        if len(run_config.optimize_space) == 0:
            raise ValueError("Key 'optimize_sace' seems to be empty.")

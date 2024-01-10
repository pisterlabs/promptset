import os
import test

import openai
import yaml
from gen_fuzzcode import gen_fuzz_code
from util import logger

import pygpt

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)


class ProfileRunner:
    def __init__(self):
        # Initialize the class
        self.profilers = [Profile()]

    def run_profile(self, code):
        # Run profile on the code
        # Return the test results
        results = []
        for profiler in self.profilers:
            result = tester.run_profile(code)
            results.append(result)


class Profile:
    def __init__(self):
        # Initialize the class
        return

    def run_profile(self, code):
        # Run profile on the code
        # Return the profile results
        return

# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-30-2023
# =============================================================================
"""This script simply collect dialogues between a 'patient' bot and a 
    'therapist' bot given the configuration from an experiment config file. 
    The config file includes parameters like decoding hyperparameters and persona."""

from typing import Dict, Any, Tuple, List
from itertools import product

import os
import pathlib

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import toml
import argparse

from easydict import EasyDict

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

import tqdm
import logging
from src.misc import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class DialogueCollector:
    """This class is used to collect dialogues between a 'patient' bot and a
        'therapist' bot given the configuration from an experiment config file.
        The config file includes parameters like decoding hyperparameters and persona."""

    def __init__(self, opt: "argparse.NameSpace") -> None:
        self.opt = opt
        self.dialogues = []
        self.load_config()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if not self.opt.debug else logging.DEBUG)

    def load_config(self) -> None:
        """Load the experiment config file from the path specified 
            in the command line arguments (`self.opt.config_path`) 
            into a dictionary (`EasyDict`)."""

        if not os.path.exists(self.opt.config_path):
            raise FileNotFoundError(f"Config file {self.opt.config_path} does not exist.")
        
        with open(self.opt.config_path, "r") as f:
            self.cfg = EasyDict(toml.load(f))

    def show_config(self) -> None:
        """Print the experiment config file to the console. This is used to 
            help check whether the config file is loaded correctly, and determine 
            if we should proceed to collect dialogues."""
    
        self.logger.info(json.dumps(self.cfg, indent=4))

    def save(self) -> "DialogueCollector":
        """Save the collected dialogues to the output directory
            specified in the command line arguments (`self.opt.output_dir`)."""

        output_dir = pathlib.Path(self.opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving dialogues to {output_dir / self.cfg.fnames.output}")
        with open(output_dir / self.cfg.fnames.output, "w") as f:
            json.dump(self.dialogues, f, indent=4)

    def collect(self) -> "DialogueCollector":
        """Driver function for collecting dialogues. It will create full 
            combinations of configurations from the experiment config file
            using different personas and hyperparameters."""

        def generate_config_stream() -> List[Tuple[Dict[str, Any]]]:
            """Generate a stream of configurations to be used for 
                generating dialogues. Each configuration is a tuple 
                of (params_therapist, params_patient, persona_therapist, 
                persona_patient)."""

            ret = []
            params_shared = self.cfg.params.shared
            for params_the, params_pat, persona_the, persona_pat in \
                product(self.cfg.params.therapist, self.cfg.params.patient, 
                            self.cfg.persona.therapist, self.cfg.persona.patient):

                params_the = {**params_shared, **params_the}
                params_pat = {**params_shared, **params_pat}

                persona_the = persona_the.text.replace("\n", "").strip()
                persona_pat = persona_pat.text.replace("\n", "").strip()
                
                ret.append((params_the, params_pat, persona_the, persona_pat))
            return ret

        def collect_single(cfgs: Tuple[Dict[str, Any]]) -> Dict[str, Any]:
            """Collect a single dialogue given the configuration tuple
                (params_therapist, params_patient, persona_therapist,
                persona_patient) for `max_num_turns` turns."""
            
            params_the, params_pat, persona_the, persona_pat = cfgs
            ret = {"meta": {"params_therapist":  params_the,
                            "params_patient":    params_pat,
                            "persona_therapist": persona_the,
                            "persona_patient":   persona_pat}, 
                   "dialogue": []}

            prompt_pat, prompt_the = persona_pat, persona_the
            self.logger.debug(f"Collecting dialogue with the following personas:")
            self.logger.debug(f"[ patient ] :: {persona_pat}")
            self.logger.debug(f"[ therapy ] :: {persona_the}")

            for _ in tqdm.trange(self.cfg.dialogue.max_num_turns, desc="Self chatting"):
                prompt = f"{prompt_pat}\nYou: \""
                response_pat = openai.Completion \
                                     .create(prompt=prompt, **params_pat) \
                                     .choices[0].text.strip().strip("\"")
                
                prompt = f"{prompt_the}\nMe: \"{response_pat}\"\nHal: \""
                response_the = openai.Completion \
                                     .create(prompt=prompt, **params_the) \
                                     .choices[0].text.strip().strip("\"")

                prompt_pat = f"{prompt_pat}\nYou: \"{response_pat}\"\nHal: \"{response_the}\""
                prompt_the = f"{prompt_the}\nMe: \"{response_pat}\"\nHal: \"{response_the}\""

                ret["dialogue"].append({"role": "patient", "text": response_pat})
                ret["dialogue"].append({"role": "therapist", "text": response_the})

                self.logger.debug(f"[ patient ] :: {response_pat}")
                self.logger.debug(f"[ therapy ] :: {response_the}")

            return ret

        # Collect dialogues
        if self.opt.ask:
            self.logger.info("The following config will be used to collect dialogues:")
            self.show_config()
            self.logger.info("Do you want to proceed? [y/n]")
            if input().lower() != "y":
                self.logger.info("Aborting...")
                return self

        for cfgs in tqdm.tqdm(generate_config_stream(), desc="Collecting dialogues for all configs"):
            for _ in tqdm.trange(self.cfg.dialogue.max_num_dialogues, desc="Collecting for one config"):
                self.dialogues.append(collect_single(cfgs))
            self.save()
        return self


def parse_args() -> "argparse.NameSpace":

    parser = argparse.ArgumentParser(
        description="Collect dialogues using self-conversation."
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str, 
        required=True,
        help="Path (including file name) to the experiment config file."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str, 
        required=True,
        help="Path (directory only) to the output file. " \
                "You should set the output data file name in config file."
    )
    parser.add_argument(
        "--ask", "-a",
        action="store_true",
        default=False,
        required=False,
        help="Whether to ask for confirmation before collecting."
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        default=False,
        required=False,
        help="Whether to print out debug logs."
    )
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    collector = DialogueCollector(parse_args())
    collector.collect()

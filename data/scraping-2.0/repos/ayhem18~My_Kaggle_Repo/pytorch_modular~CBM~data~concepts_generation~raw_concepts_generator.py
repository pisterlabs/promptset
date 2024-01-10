"""
DISCLAIMER: This script is mainly a personalized version of the concepts generation process suggested by the authors of
the 'Label-Free Concept Bottleneck Models' paper.

The original code can be found via this link: "https://github.com/Trustworthy-ML-Lab/Label-free-CBM"
"""
import re

import openai

from typing import Iterable, Dict, Union, Set
from abc import ABC, abstractmethod


class _RawConceptGenerator(ABC):
    @abstractmethod
    def generate_concepts(self, classes: Iterable[str]) -> Dict[str, Iterable[str]]:
        """
        Args:
            classes: The classes for which concepts should be generated

        Returns: a dictionary mapping each class to its concepts
        """
        pass


# these are the default prompts suggested by the 'Free-Label CBM' paper
PROMPTS = {
    "important": "List the most important features for recognizing something as a \"goldfish\":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a \"beerglass\":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a \"{}\":",
    "superclass": "Give superclasses for the word \"tench\":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word \"beer glass\":\n\n-glass\n-container\n-object\n\nGive superclasses for the word \"{}\":",
    "around": "List the things most commonly seen around a \"tench\":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a \"beer glass\":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a \"{}\":"
}


class RawChatGptConceptGenerator(_RawConceptGenerator):
    default_prompt_name = 'default'

    def __init__(self,
                 prompts: Union[Dict[str, str], Iterable[str], str] = None,
                 iterations_per_label: int = 2,
                 temperature: float = 0.2
                 ):
        # set the default prompts argument
        prompts = PROMPTS if prompts is None else prompts

        if not isinstance(prompts, (Dict, str, Iterable[str])):
            raise TypeError(f"The class expects to be either {Dict}, {str} or {Iterable}. Found: {type(prompts)}")

        # convert all the prompts into a dictionary
        if isinstance(prompts, str):
            prompts = {self.default_prompt_name: prompts}

        elif isinstance(prompts, Dict):
            prompts = prompts

        elif isinstance(prompts, Iterable):
            prompts = dict([(index, pt) for index, pt in enumerate(prompts)])

        self.default_prompts = prompts
        self.iterations_per_label = iterations_per_label
        self.temperature = temperature

    def concepts_by_prompts_and_cls(self, prompt: str, cls: str):

        features = set()

        # go through the generation process several times (at least twice) for better results
        for _ in range(self.iterations_per_label):
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt.format(cls),
                temperature=self.temperature,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # process the response
            intermediate_feats = response["choices"][0]["text"].split("\n-")
            # remove the extra spaces
            intermediate_feats = [re.sub("\n", "", feat).strip() for feat in intermediate_feats]
            # remove any empty strings
            intermediate_feats = set([feat for feat in intermediate_feats if len(feat) > 0])

            # add the intermediate features to the results
            features.update(intermediate_feats)

        return features

    def generate_concepts(self,
                          classes: Iterable[str],
                          prompt_type: str = 'all',
                          by_type: bool = False) -> Union[Dict[str, Dict[str, str]], Dict[str, Set[str]]]:

        ptypes = list(self.default_prompts.keys()) if prompt_type == 'all' else prompt_type

        # make sure the passed prompt types  belong to the default ones
        for pt in ptypes:
            if pt not in self.default_prompts:
                raise ValueError(f"prompt types are expected to belong to {ptypes}\n"
                                 f"Found: {pt}")

        label_concept_dict = {}

        for cls in classes:
            label_concept_dict[cls] = ({} if by_type else set())

            cls_value = label_concept_dict[cls]

            for pt in ptypes:
                # initialize the prompt
                current_prompt = self.default_prompts[pt]

                # generate the features by class and by prompt type
                features = self.concepts_by_prompts_and_cls(prompt=current_prompt, cls=cls)

                if by_type:
                    # in this case, cls_value will be a dictionary that maps pt to the concepts associated with it
                    cls_value[pt] = sorted(list(features))
                else:
                    # in this case cls_value[pt] is a set and the ones
                    cls_value.update(features)

            # make sure to convert the 'set' to a 'list' object
            label_concept_dict[cls] = sorted(list(cls_value))
        return label_concept_dict

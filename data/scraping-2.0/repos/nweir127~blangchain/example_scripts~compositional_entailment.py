import asyncio
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from langchain import PromptTemplate
from tqdm import tqdm
from transformers import HfArgumentParser
from src.generators.openai_gpt import SimplePromptOpenAIGenerator, JSONItemGenerator
from src.utils import flatten

PROMPT = """For each of the following questions, we have asked a reasoning system to give an answer and justify its decision in the form of "BECAUSE ____ AND ____ IT FOLLOWS THAT _____.

Your job is to help debug the model's line of reasoning. Are the individual facts that it used correct? 
For each fact in the answer, provide a score between 1 and 5 reflecting how often the statement holds true in the wild.
1 is "never true"
2 is "rarely true"
3 is "sometimes true"
4 is "usually true"
5 is "always true"

Your output format is a serialized json on a single line with the format {{"<fact1>": <fact1_score>, "fact2": <fact2_score>}} and nothing else.  


QUESTION 1:
Which is an example of conduction? (A) a space heater turned on (B) water boiling on the stove (C) sunlight shining through the window (D, CORRECT) a metal spoon warming in a pot of hot soup

SYSTEM RESPONSE 1:
BECAUSE a space heater is a kind of electrical device
AND an electrical device turns on / off
IT FOLLOWS THAT a space heater turned on is an example of conduction

DEBUG 1:
{{"a space heater is a kind of electrical device": 5, "an electrical device turns on / off": 4}}

QUESTION 2:
How do the spines of a cactus help it survive? (A) Spines help the cactus get moisture. (B) Spines anchor the cactus in the ground. (C, CORRECT) Spines protect the cactus from animals. (D) Spines support the stems and branches of the cactus.

RESPONSE 2:
BECAUSE the spines of a cactus help it survive by keeping predators away
AND keeping predators away helps a cactus get moisture
IT FOLLOWS THAT the spines of a cactus help it survive by the cactus get moisture

DEBUG 2:
{{"the spines of a cactus help it survive by keeping predators away": 5, "keeping predators away helps a cactus get moisture": 2}}

QUESTION 3:
{question}

RESPONSE 3:
{response}

DEBUG 3:
"""


class FactScoreGenerator(SimplePromptOpenAIGenerator, JSONItemGenerator):
    def __init__(self, **kwargs):
        prompt_template = PromptTemplate.from_template(PROMPT)
        super(FactScoreGenerator, self).__init__(prompt_template=prompt_template, **kwargs)

    async def score_facts(self, question_text: str, premises: List[str], hypothesis: str, n_outputs=3):
        inputs = [dict(
            question=question_text,
            response="BECAUSE {} ".format(premises[0]) + \
                     ' '.join('AND {}'.format(p) for p in premises[1:]) + \
                     "IT FOLLOWS THAT {}".format(hypothesis)
        )]

        generation = (await self.agenerate(inputs, n=3))[0]
        scores = [self.postprocess_generation(g) for g in generation]
        min_scores = {k: min(sc_i[0][k] for sc_i in scores) for k in scores[0][0]}

        return min_scores


if __name__ == "__main__":
    @dataclass
    class CompositionalEntailmentArguments:
        model: str = field(default="chatgpt")
        out_dir: str = field(default='tmp')
        max_batches: int = field(default=10000)


    (args,) = HfArgumentParser(CompositionalEntailmentArguments).parse_args_into_dataclasses()
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    ### replace with dataset iterator
    ### because it's async, you can run this in batches

    dataset = [
        dict(id="AIMS_2009_4_19-B",
             premises=["a glacier will not create soil by carrying plants down mountains and to oceans",
                       "creating something is the opposite of carrying something"],
             hypothesis="a glacier will create soil by plants down mountains and to oceans",
             question="A glacier is a slow moving river of ice. How does a glacier help create soil? \
             (A, CORRECT) It scrapes small particles off large rocks. \
             (B) It carries plants down mountains and to oceans. \
             (C) It melts and becomes part of streams and rivers. \
             (D) It freezes small particles of dirt to form large rocks.")
    ]

    scorer = FactScoreGenerator(model=args.model)
    all_outputs = []
    batch_size = 5
    for batch_idx, i in enumerate(tqdm(range(0, len(dataset), batch_size))):
        jobs = [
            scorer.score_facts(premises=inp['premises'], hypothesis=inp['hypothesis'], question_text=inp['question'])
            for inp in dataset[i:i + batch_size]]


        async def _run():
            all_answers = await asyncio.gather(*jobs)
            return all_answers


        all_outputs.extend(asyncio.run(_run()))
        if batch_idx == args.max_batches: break

    breakpoint()

    Path(f"{args.out_dir}/{args.dataset}").mkdir(parents=True, exist_ok=True)
    # save outputs in a manner that makes sense for you
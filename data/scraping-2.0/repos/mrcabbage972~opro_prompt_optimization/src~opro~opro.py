import logging
import os
from typing import List

import openai
from dotenv import load_dotenv
from gsm8k import get_dataset
from schema import ProblemExample
from schema import PromptExample
from tqdm import tqdm

from src.opro.prompt_generation import generate_prompt_candidates
from src.opro.prompt_scoring import score_prompt_candidates
from src.opro.settings import MAX_ITER
from src.opro.settings import MAX_PROMPT_CANDIDATES
from src.opro.settings import MAX_TEST_EXAMPLES
from src.opro.settings import MAX_TRAIN_EXAMPLES

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def seed_prompt_examples(demo_examples: List[ProblemExample], test_examples: List[ProblemExample]) \
        -> List[PromptExample]:
    default_prompt_examples = ["Let’s solve the problem."]
    return score_prompt_candidates(default_prompt_examples, demo_examples, test_examples)


def update_prompt_examples(previous_prompt_examples: List[PromptExample],
                           scored_prompt_candidates: List[PromptExample]) -> List[PromptExample]:
    all_candidates = previous_prompt_examples + scored_prompt_candidates
    return sorted(all_candidates, key=lambda x: x.score, reverse=True)[:MAX_PROMPT_CANDIDATES]


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    train_examples = get_dataset('train')[:MAX_TRAIN_EXAMPLES]
    test_examples = get_dataset('test')[:MAX_TEST_EXAMPLES]

    prompt_examples = seed_prompt_examples(train_examples[:1], test_examples)

    for _ in tqdm(range(MAX_ITER), desc='Optimization iteration'):
        prompt_candidates = generate_prompt_candidates(prompt_examples, train_examples)
        scored_prompt_candidates = score_prompt_candidates(prompt_candidates, train_examples, test_examples)
        prompt_examples = update_prompt_examples(prompt_examples, scored_prompt_candidates)
        best_score = max([x.score for x in prompt_examples])
        LOGGER.info('Current best score: %s', best_score)


if __name__ == '__main__':
    main()

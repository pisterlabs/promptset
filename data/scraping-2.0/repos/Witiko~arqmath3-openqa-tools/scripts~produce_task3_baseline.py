import argparse
import csv
import logging
import os
from pathlib import Path
import re

import openai
from pv211_utils.arqmath.entities import ArqmathQueryBase as Topic
from pv211_utils.arqmath.loader import load_queries as load_topics
from tqdm import tqdm


COMPLETION_PARAMETERS = {
    'engine': 'text-davinci-002',
    'max_tokens': 570,
    'temperature': 0.7,
}

LOGGER = logging.getLogger(__name__)


def produce_answer(topic: Topic, max_answer_length: int = 1200) -> str:
    prompt = f'Q: {topic.title}\n\n{topic.body}\n\n'
    while True:
        answer = openai.Completion.create(
            prompt=prompt,
            **COMPLETION_PARAMETERS
        )
        choice, = answer['choices']
        if choice['finish_reason'] != 'stop':
            continue
        text = choice['text']
        text = re.sub('.*A:', '', text).strip()
        if len(text) <= max_answer_length:
            break
    return text


def produce_results(output_file: Path, year: int) -> None:
    topics = load_topics('text+latex', year=year)
    with output_file.open('wt', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for topic_id, topic in tqdm(topics.items()):
            answer = produce_answer(topic)
            row = (f'A.{topic_id}', '1', '1.0', 'GPT3', 'GPT3', answer)
            csv_writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    openai.api_key = os.getenv('OPENAI_API_KEY')

    parser = argparse.ArgumentParser(description='Produce GPT-3 baseline results for ARQMath Task 3')
    parser.add_argument('-out', help='Output result file in ARQMath format for ARQMath Task 3', required=True)
    parser.add_argument('-year', help='The year of the topics that we produce the results for', required=True)

    args = parser.parse_args()
    output_file = Path(args.out)
    year = int(args.year)

    produce_results(output_file, year)


if __name__ == '__main__':
    main()

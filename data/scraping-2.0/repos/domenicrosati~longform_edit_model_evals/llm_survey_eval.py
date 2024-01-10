import argparse
import json
import os

import openai

from src.llm_survey import get_survey_results

parser = argparse.ArgumentParser()
parser.add_argument('--sample-dir', type=str)
parser.add_argument('--sample-type', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--n-shots', type=int, default=0)
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')

args = parser.parse_args()

openai.api_key = os.environ['OPENAI_API_KEY']


def get_samples_from_dir(dir_path):
    samples = []
    for file_name in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_name), 'r') as f:
            samples.append(json.load(f))
    return samples


if __name__ == '__main__':
    sample_dir = args.sample_dir
    sample_type = args.sample_type
    model = args.model

    print(sample_type)
    print(model)

    # load the samples
    samples = get_samples_from_dir(sample_dir)

    # get the results
    results = get_survey_results(
        samples,
        model=model,
        n_shots=args.n_shots
    )

    overall_scores = {}
    for sample_id, sample_results in results.items():
        for label, score in sample_results.items():
            if label not in overall_scores:
                overall_scores[label] = []
            overall_scores[label].extend(score)
    # print the overall scores
    print(len(overall_scores['new_fact_main_passage']))
    for label, scores in overall_scores.items():
        print(label, sum([s for s in scores if s]) / len([s for s in scores if s]))
    print()

    with open(f'./results/broken_out_survey_{sample_type}_{model}_shots_{args.n_shots}.json', 'w') as f:
        json.dump(results, f)

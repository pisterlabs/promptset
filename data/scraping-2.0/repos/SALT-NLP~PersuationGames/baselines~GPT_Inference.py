import os
from collections import defaultdict

import openai
import json
import argparse

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument("--prompt_template", default="baselines/prompt.txt", type=str)
parser.add_argument("--model", default="text-davinci-002")
parser.add_argument("--t", default=0, type=float)
parser.add_argument("--top_p", default=1.0, type=float)
parser.add_argument("--max_tokens", default=1024, type=int)
parser.add_argument("--n", default=1, type=int)
parser.add_argument("--echo", action='store_true', help="whether to include prompt in the outputs")
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--dataset", nargs='+', default=('Ego4D',), type=str, help="Name of dataset, Ego4D or Youtube")
parser.add_argument("--splits", nargs="+", default=("dev", ), type=str)
parser.add_argument("--output_dir", default="out/GPT-3", type=str)
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

Strategies = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def get_response(prompt, output_file):
	response = openai.Completion.create(
		model = args.model,
		prompt = prompt,
		temperature = args.t,
		top_p = args.top_p,
		max_tokens = args.max_tokens,
		n = args.n,
		logprobs = 5 if args.verbose else None,
		echo = args.echo)
	if args.verbose:
		with open(f"{output_file}.json", 'w') as f:
			f.write(json.dumps(response, indent=4))
			f.write('\n')
	# print(response)
	# response.to_dict_recursive()
	text = [response['choices'][i]['text'] for i in range(args.n)]
	return text

if __name__ == "__main__":
	with open(args.prompt_template, "r") as f:
		prompt_template = ''.join(f.readlines())
	# print(prompt_template)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	for split in args.splits:
		df = None
		csv_path = os.path.join(args.output_dir, f'predictions_{split}.csv')
		if args.overwrite or not os.path.exists(csv_path):
			with open(os.path.join('data', args.dataset[0], f'{split}.json'), 'r') as f:
				games = json.load(f)
			data = defaultdict(list)
			id = 0
			for game in games:
				# print(f"{game['EG_ID']}_{game['Game_ID']}")
				for record in game['Dialogue']:
					id += 1
					# if id < 37: continue
					# if id > 40: break
					utterance = record["utterance"]
					prompt = prompt_template.replace('$utterance$', utterance)
					print(prompt)
					pred = get_response(prompt, os.path.join(args.output_dir, f"raw_{game['Game_ID']}_{record['Rec_Id']}"))
					for key, val in record.items():
						data[key].append(val)
					data['prediction'].append(pred)
			# print(f"{record['Rec_Id']},{record['speaker']},{record['timestamp']},{record['utterance']},{record['annotation']},{pred}")
			df = pd.DataFrame.from_dict(data)
			df.to_csv(os.path.join(args.output_dir, f'predictions_{split}.csv'))

		df = pd.read_csv(csv_path)

		result = {}
		averaged_f1 = 0.0
		all_correct = [1] * len(df['prediction'])
		# print(all_correct)
		for strategy in Strategies:
			preds = [1 if strategy in prediction else 0 for prediction in df['prediction']]
			labels = [1 if strategy in annotation else 0 for annotation in df['annotation']]
			# print(preds)
			# print(labels)
			for i in range(len(all_correct)):
				if preds[i] != labels[i]:
					all_correct[i] = 0
			# print(all_correct)
			result[strategy] = {
				'f1': f1_score(y_true=labels, y_pred=preds),
				'precision': precision_score(y_true=labels, y_pred=preds),
				'recall': recall_score(y_true=labels, y_pred=preds),
				'accuracy': accuracy_score(y_true=labels, y_pred=preds),
				'report': classification_report(y_true=labels, y_pred=preds),
			}
			averaged_f1 += result[strategy]['f1']

		result['overall_accuracy'] = sum(all_correct) / len(all_correct)
		result['averaged_f1'] = averaged_f1 / len(Strategies)

		filename = os.path.join(args.output_dir, f'results_{split}.json')
		with open(filename, 'w') as f:
			json.dump(result, f)

		# beautiful print results
		with open(os.path.join(args.output_dir, f"results_{split}_beaut.txt"), 'w') as f:

			for strategy in Strategies:
				f.write(f"{result[strategy]['f1'] * 100:.1f}\t")
			f.write(f"{result['averaged_f1'] * 100:.1f}\t{result['overall_accuracy'] * 100:.1f}\n")

			for strategy in Strategies:
				report = result[strategy]['report']
				result[strategy].pop('report')
				f.write(f"{strategy}\n")
				json.dump(result[strategy], f, indent=4)
				f.write(report)
				f.write("\n")
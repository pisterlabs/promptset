from utils import load_json_dataset
from squad_eval import evaluate_predictions
from datasets import load_dataset
import argparse

from electra_for_qa import ElectraForQA, SQuAD2PredictionDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import scipy

import openai
import time
from timeit import default_timer
import os
import jsonlines
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration

import json
from align.inference import Inferencer
import tiktoken


def get_electra_model(device, ckpt, result_path, batch_size, **kwargs):
    model = ElectraForQA.load_from_checkpoint(ckpt)
    trainer = Trainer(
        accelerator='gpu',
        devices=[device],
        default_root_dir=result_path,
    )

    def make_predictions(dataset, name):
        print(f'Electra {name}')
        eval_dataset = SQuAD2PredictionDataset(dataset)
        loader = DataLoader(eval_dataset, batch_size=batch_size)
        predictions = trainer.predict(model, loader)

        answers = {}
        na_prob = {}
        for batch in predictions:
            for sample_id, prediction, no_ans_logit in zip(
                batch['sample_ids'],
                batch['predictions'],
                batch['answerable_logits']
            ):
                answers[sample_id] = prediction[0][0]
                na_prob[sample_id] = scipy.special.expit(-no_ans_logit)
        return answers, na_prob

    return make_predictions


def get_gpt_model(wait_time, result_path, max_length, **kwargs):
    prompt = 'Find the answer to the question from the given context. '\
        + 'When the question cannot be answered with the given context, say "unanswerable". '\
        + 'Just say the answer without repeating the question.\n'\
        + 'Context: {context}\nQuestion: {question}\nAnswer:'
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def generate_prompt(context, question):
        input_text = prompt.format(context=context, question=question)
        input_tokens = encoding.encode(input_text)
        if len(input_tokens) <= max_length:
            return input_text
        context_tokens = encoding.encode(context)
        context_len = len(context_tokens) - (len(input_tokens) - max_length)
        short_context = encoding.decode(
            context_tokens[:context_len], errors='ignore')
        return prompt.format(context=short_context, question=question)

    def parse_answer(answer):
        no_ans_keywords = [
            'unanswerable',
            'no answer'
            'context does not provide an answer'
        ]
        if any(k in answer for k in no_ans_keywords):
            return ''
        return answer

    raw_results = os.path.join(result_path, 'raw_results')
    os.makedirs(raw_results, exist_ok=True)

    def make_predictions(dataset, name):
        cache_file = os.path.join(raw_results, f'{name}.jsonl')
        completed = set()
        try:
            with jsonlines.open(cache_file, 'r') as results:
                for result in results:
                    completed.add(result['id'])
        except FileNotFoundError:
            pass
        with jsonlines.open(cache_file, 'a') as results:
            for sample in tqdm(dataset, desc=f'GPT {name}'):
                if sample['id'] in completed:
                    continue
                start = default_timer()
                prompt = generate_prompt(sample['context'], sample['question'])
                result = None
                exception_time = wait_time
                exception_count = 0
                while result is None:
                    try:
                        result = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": prompt},
                            ]
                        )
                    except (openai.error.RateLimitError, openai.error.APIError):
                        exception_count += 1
                        if exception_count <= 10:
                            time.sleep(exception_time)
                            exception_time *= 2
                        else:
                            raise
                results.write({'id': sample['id'], 'result': result})
                end = default_timer()
                time.sleep(max(0, wait_time - (end - start)))

        with jsonlines.open(cache_file, 'r') as results:
            predictions = {
                result['id']: parse_answer(
                    result['result']['choices'][0]['message']['content'])
                for result in results
            }
        return predictions, None

    return make_predictions


def get_flan_t5_model(model, device, max_length, **kwargs):
    tokenizer = T5Tokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)
    model.to(f'cuda:{device}')

    def make_predictions(dataset, name):
        predictions = {}
        for sample in tqdm(dataset, desc=f'FLAN T5 {name}'):
            context = sample['context']
            question = sample['question']

            input_seq = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            inputs = tokenizer(
                input_seq,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            )
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                max_new_tokens=40,
            )
            candidate = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            if not candidate or candidate == 'unanswerable':
                predictions[sample['id']] = ''
            else:
                predictions[sample['id']] = candidate

        return predictions, None

    return make_predictions


class AlignVerifier:
    def __init__(self, align_ckpt, align_model, device) -> None:
        self.align = Inferencer(
            ckpt_path=align_ckpt, model=align_model, device=device, verbose=False)
        self.align.nlg_eval_mode = 'nli_sp'
        self.qa2d = None
        self.device = device

    def get_no_prob(self, context, question, candidate):
        if not candidate:
            return 1.
        if self.qa2d is None:
            from data_utils.generate_training_data import QA2D
            self.qa2d = QA2D(device=self.device, verbose=False)
        hypo = self.qa2d.generate([question], [candidate])
        return 1. - self.align.nlg_eval([context], hypo)[1][0].item()

    def get_no_prob_concat(self, context, question, candidate):
        if not candidate:
            return 1.
        hypo = question + ' ' + candidate
        return 1. - self.align.inference([context], [hypo])[1][0].item()


def combine_ace_whqa_subsets(subsets):
    data = []
    for name, dataset in subsets:
        for sample in dataset:
            sample = {**sample}
            sample['id'] = f'{name}/{sample["id"]}'
            data.append(sample)
    return data


def combine_ace_whqa_predictions(subsets):
    all_predictions = {}
    for name, predictions in subsets.items():
        all_predictions.update({
            f'{name}/{id}': val for id, val in predictions.items()
        })
    return all_predictions


ZEROSHOT_DATASETS = [
    {
        'name': 'ace_whqa_has_answer',
        'path': 'ace_whqa/ACE-whQA-has-answer.json'
    },
    {
        'name': 'ace_whqa_IDK_competitive',
        'path': 'ace_whqa/ACE-whQA-IDK-competitive.json'
    },
    {
        'name': 'ace_whqa_IDK_non_competitive',
        'path': 'ace_whqa/ACE-whQA-IDK-non-competitive.json'
    },
    {
        'name': 'simplified_nq',
        'path': 'simplified_nq.json'
    },
]

MODELS = [
    {'name': 'electra', 'load_model': get_electra_model},
    {'name': 'gpt', 'load_model': get_gpt_model},
    {'name': 'flan_t5', 'load_model': get_flan_t5_model},
]


def run_benchmark(args):
    datasets = [
        ('squad_v2', load_dataset('squad_v2')['validation']),
    ]
    for dataset_dict in ZEROSHOT_DATASETS:
        path = os.path.join(args.data_path, dataset_dict['path'])
        datasets.append((dataset_dict['name'], load_json_dataset(path)))

    # all answers without applying "not answerable" threshold
    all_predictions = {}
    all_na_prob = {}
    na_threshold = {}

    for model_dict in MODELS:
        model_name = model_dict['name']
        result_path = os.path.join(args.result_path, model_name)
        evaluation_path = os.path.join(result_path, 'evaluation')
        prediction_path = os.path.join(result_path, 'predictions')
        os.makedirs(evaluation_path, exist_ok=True)
        os.makedirs(prediction_path, exist_ok=True)
        model_args = {
            key[len(model_name)+1:]: val
            for key, val in vars(args).items()
            if key.startswith(model_name + '_')
        }
        model = model_dict['load_model'](
            device=args.device,
            result_path=result_path,
            **model_args
        )
        for dataset_name, dataset in datasets:
            predictions_file = os.path.join(
                prediction_path,
                f'{dataset_name}_predictions.json'
            )
            na_prob_file = os.path.join(
                prediction_path,
                f'{dataset_name}_na_prob.json'
            )

            evaluation_file = os.path.join(
                evaluation_path, f'{dataset_name}.json')
            
            if os.path.exists(predictions_file):
                print(f'Reading cached results {predictions_file}')
                with open(predictions_file, 'r', encoding='utf8') as file:
                    predictions = json.load(file)
                na_prob = None
                try:
                    with open(na_prob_file, 'r', encoding='utf8') as file:
                        na_prob = json.load(file)
                except FileNotFoundError:
                    pass
            else:
                predictions, na_prob = model(dataset, dataset_name)

                with open(predictions_file, 'w', encoding='utf8') as file:
                    json.dump(predictions, file)
                if na_prob is not None:
                    all_na_prob[(model_name, dataset_name)] = na_prob
                    with open(na_prob_file, 'w', encoding='utf8') as file:
                        json.dump(na_prob, file)

            all_predictions[(model_name, dataset_name)] = predictions

            threshold = na_threshold.get(model_name)
            evaluation_results = evaluate_predictions(
                dataset,
                predictions,
                na_prob,
                threshold,
                name=f'{model_name} {dataset_name}'
            )
            print(model_name, dataset_name)
            print(evaluation_results)

            with open(evaluation_file, 'w', encoding='utf8') as file:
                json.dump(evaluation_results, file)

            if dataset_name == 'squad_v2' and 'best_f1_thresh' in evaluation_results:
                na_threshold[model_name] = evaluation_results['best_f1_thresh']
        del model

    align_verifier = AlignVerifier(
        args.align_ckpt,
        args.align_model,
        f'cuda:{args.device}'
    )
    for model_dict in MODELS:
        base_model_name = model_dict['name']
        model_name = 'align_' + base_model_name
        result_path = os.path.join(args.result_path, model_name)
        evaluation_path = os.path.join(result_path, 'evaluation')
        prediction_path = os.path.join(result_path, 'predictions')
        os.makedirs(evaluation_path, exist_ok=True)
        os.makedirs(prediction_path, exist_ok=True)
        for dataset_name, dataset in datasets:
            model_predictions = all_predictions[(
                base_model_name, dataset_name)]
            align_na = {}
            for sample in tqdm(dataset, desc=f'align {model_name} {dataset_name}'):
                align_na[sample['id']] = align_verifier.get_no_prob_concat(
                    sample['context'],
                    sample['question'],
                    model_predictions[sample['id']]
                )

            all_predictions[(model_name, dataset_name)] = model_predictions
            all_na_prob[(model_name, dataset_name)] = align_na

            na_prob_file = os.path.join(
                prediction_path,
                f'{dataset_name}_na_prob.json'
            )
            evaluation_file = os.path.join(
                evaluation_path,
                f'{dataset_name}.json'
            )

            with open(na_prob_file, 'w', encoding='utf8') as file:
                json.dump(align_na, file)

            threshold = na_threshold.get(model_name)
            evaluation_results = evaluate_predictions(
                dataset,
                model_predictions,
                align_na,
                threshold,
                name=f'{model_name} {dataset_name}'
            )
            print(model_name, dataset_name)
            print(evaluation_results)

            with open(evaluation_file, 'w', encoding='utf8') as file:
                json.dump(evaluation_results, file)

            if dataset_name == 'squad_v2':
                na_threshold[model_name] = evaluation_results['best_f1_thresh']

    ace_whqa_subsets = [
        (name, dataset) for name, dataset in datasets
        if name.startswith('ace_whqa_')
    ]
    ace_whqa_all = combine_ace_whqa_subsets(ace_whqa_subsets)
    for model_dict in MODELS:
        models = [model_dict['name'], 'align_' + model_dict['name']]
        for model_name in models:
            result_path = os.path.join(args.result_path, model_name)
            evaluation_path = os.path.join(result_path, 'evaluation')
            os.makedirs(evaluation_path, exist_ok=True)
            combined_predictions = combine_ace_whqa_predictions({
                subset: all_predictions[(model_name, subset)]
                for subset, _ in ace_whqa_subsets
            })
            if all(((model_name, subset) in all_na_prob for subset, _ in ace_whqa_subsets)):
                combined_na_prob = combine_ace_whqa_predictions({
                    subset: all_na_prob[(model_name, subset)]
                    for subset, _ in ace_whqa_subsets
                })
            else:
                combined_na_prob = None

            evaluation_file = os.path.join(
                evaluation_path,
                f'ace_whqa_all.json'
            )

            threshold = na_threshold.get(model_name)
            evaluation_results = evaluate_predictions(
                ace_whqa_all,
                combined_predictions,
                combined_na_prob,
                threshold,
                name=f'{model_name} ace_whqa_all'
            )
            print(model_name, 'ace_whqa_all')
            print(evaluation_results)

            with open(evaluation_file, 'w', encoding='utf8') as file:
                json.dump(evaluation_results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="qa_results"
    )
    parser.add_argument(
        "--electra_ckpt",
        type=str,
        default='align_qa/electra_squad2/lightning_logs/version_0/checkpoints/epoch=1-val_loss_epoch=1.930589.ckpt'
    )
    parser.add_argument(
        "--electra_batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--gpt_wait_time",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--gpt_max_length",
        type=int,
        default=2040
    )
    parser.add_argument(
        "--flan_t5_model",
        type=str,
        default="google/flan-t5-xl"
    )
    parser.add_argument(
        "--flan_t5_max_length",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--align_ckpt",
        type=str,
        default='checkpoints/more-qa-scale-loss/roberta-large/roberta-large_no_mlm_full-dataset_500000_32x4x1_final.ckpt'
    )
    parser.add_argument(
        "--align_model",
        type=str,
        default='roberta-large'
    )
    parser.add_argument(
        "--device",
        type=int,
        default=6
    )
    args = parser.parse_args()
    run_benchmark(args)

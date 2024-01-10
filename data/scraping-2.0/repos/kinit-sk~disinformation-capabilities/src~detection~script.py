from detectors.zerogpt import zerogpt_det
from detectors.simpleai_detector import simpleai_det
from detectors.openai_detector import openai_det
from detectors.longformer_detector import longformer_det
from detectors.llmdet import llmdet_det
from detectors.grover_detector import grover_det
from detectors.gptzero import gptzero_det
from detectors.gltr import get_results
from detectors.detector import detector_det
import argparse
import pandas as pd
import nltk

nltk.download('punkt')


def load_data(path='../../data/mgt_detection.csv'):
    data = pd.read_csv(path)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection experiments')
    parser.add_argument('--model', type=str, default='all',
                        help='name of the model', required=True)
    parser.add_argument('--model_path', type=str,
                        default='../../data/models', help='path to the model')
    args = parser.parse_args()

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)

    if args.model == 'all':
        detectors = [
            'gpt2-finetuned-en3-all',
            'electra-small-discriminator-finetuned-en3-all',
            'bert-base-multilingual-cased-finetuned-en3-all',
            'roberta-large-openai-detector-finetuned-en3-all',
            'xlm-roberta-large-finetuned-en3-all',
            'electra-large-discriminator-finetuned-en3-all',
            'mdeberta-v3-base-finetuned-en3-all',
            'gpt2-medium-finetuned-en3-all',
            'mGPT-finetuned-en3-all',
            'opt-iml-max-1.3b-finetuned-en3-all',
            'electra-large-discriminator-finetuned-en3-gpt-3.5-turbo',
            'electra-large-discriminator-finetuned-en3-opt-iml-max-1.3b',
            'electra-large-discriminator-finetuned-en3-text-davinci-003',
            'electra-large-discriminator-finetuned-en3-vicuna-13b',
            'electra-large-discriminator-finetuned-en3-gpt-4'
        ]

        zerogpt_det(data=DATA)
        simpleai_det(data=DATA)
        openai_det(data=DATA)
        longformer_det(data=DATA)
        llmdet_det(data=DATA)
        grover_det(data=DATA)
        gptzero_det(data=DATA)
        get_results(data=DATA)
        for detector in detectors:
            detector_det(data=DATA, name=detector,
                         model_path=args.model_path,
                         output_name=f'../../data/results/{detector}.csv')
    else:
        if args.model == 'zerogpt':
            zerogpt_det(data=DATA)
        elif args.model == 'simpleai':
            simpleai_det(data=DATA)
        elif args.model == 'openai':
            openai_det(data=DATA)
        elif args.model == 'longformer':
            longformer_det(data=DATA)
        elif args.model == 'llmdet':
            llmdet_det(data=DATA)
        elif args.model == 'grover':
            grover_det(data=DATA)
        elif args.model == 'gptzero':
            gptzero_det(data=DATA)
        elif args.model == 'gltr':
            get_results(data=DATA)
        else:
            detector_det(data=DATA, name=args.model,
                         model_path=args.model_path,
                         output_name=f'../../data/results/{args.model}.csv')

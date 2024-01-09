import sys
import os
import openai
from transformers import AutoTokenizer, AutoConfig
from unidecode import unidecode
import re
import jsonlines
import torch
from tqdm.auto import tqdm
import json

CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, '../quote_detection/other_platforms/src'))
sys.path.insert(0, os.path.join(here, '../quote_attribution/other_platforms/gpt3-juno-finetuning'))
sys.path.insert(0, os.path.join(here, '../quote_attribution/other_platforms/span-detection-approaches'))

sep = '\n\n##\n\n'
end = ' END'
prohibited_tokens = {
    str(k):-100 for k in [464, 6827, 373, 531, 416, 1212, 9577, 59, 77, 1212, 2723, 318, 422, 33706, 25]
}

ACCEPTED_ATTRIBUTION_MODELS = [
    'curie:ft-isi-nlp:sep-training-set-base-2022-12-02-01-29-12',  # curie (expensive)
    'babbage:ft-university-of-southern-california-2023-01-04-22-43-29',  # babbage without nones
    'babbage:ft-isi-nlp-2023-01-12-06-58-08',  # babbage without nones replica
    'babbage:ft-isi-nlp-2023-01-11-10-35-02',  # babbage with coreference
    'babbage:ft-isi-nlp-2023-01-07-09-33-35',  # babbage with nones
    'babbage:ft-isi-nlp-2023-01-12-18-30-08',  # babbage with some nones
    'quote_attribution__bigbird-roberta-large',
    'quote_attribution__bigbird-roberta-base',
    'quote_attribution__gpt-neo',
]

KEYS = {
    'isi': "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",
    'usc': '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',
    'personal': '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',
    'stanford': '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',
}

class OpenAIModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, prompt):
        try:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                n=1,
                max_tokens=10,
                stop='END',
                logit_bias=prohibited_tokens
            )
            model_output = response.to_dict_recursive()['choices'][0]['text'].strip()

            return model_output
        except Exception as e:
            print('attribution error: %s' % str(e))
            return None

    def prepare_sent(self, p):
        return p

    def process_output(self, a, p):
        return a


class OpenAIAttributionDataset():
    def __init__(self, tokenizer, max_len=2040):
        self.prompt_template = '"""%s""".\n\nTo which source can we attribute this sentence:\n\n"""%s"""\n\n##\n\n'
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_one_doc(self, one_doc):
        """
        Makes a prompt for OpenAI fine-tuned model. Here, we're not training. We expect `one_doc_df`
        to have the following columns:

        * `sent`     if detection has not been run, and
        * `sent`, `is_quote` if detection has been run.
        """

        doc_sents = list(map(lambda x: cleanhtml(x['sent']), one_doc))
        article = ' '.join(doc_sents)

        all_prompts = []
        for sent in one_doc:
            sent_text = cleanhtml(sent['sent'])
            num_toks = len(self.tokenizer.encode(sent_text))
            if (len(sent_text) > 2) and sent.get('is_quote') and (num_toks < self.max_len):
                prompt = self.prompt_template % (article, sent_text)
            else:
                prompt = None
            all_prompts.append({'prompt': prompt})

        return all_prompts


class QADatasetWrapper():
    def __init__(self, qa_dataset, tokenizer, collator, device=None):
        self.qa_dataset = qa_dataset
        self.collator = collator
        self.device = device or get_device()
        self.tokenizer = tokenizer

    def process_one_doc(self, doc):
        return self.qa_dataset.process_one_doc(doc)

    def prepare_sent(self, input_packet):
        output = {}
        cols = ['input_ids', 'token_type_ids']
        for col in cols:
            output[col] = torch.tensor(input_packet[col]).unsqueeze(dim=0).to(self.device)
        return output

    def process_output(self, attribution, input_packet):
        start_token, end_token = list(map(lambda x: x.argmax(), attribution))
        start_token, end_token = min(start_token, end_token), max(start_token, end_token)
        span = input_packet['input_ids'][0, start_token: end_token + 1].detach().cpu().numpy()
        return self.tokenizer.decode(span)


def get_attribution_model_type(model_name):
    if 'babbage' in model_name or 'curie' in model_name:
        model_type = 'openai'
    else:
        model_type = 'hf'
    if 'isi-nlp' in model_name:
        key = 'isi'
    else:
        key = 'usc'
    return model_type, key



def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--detection_model', type=str, )
    parser.add_argument('--detection_config', type=str, default=None)
    parser.add_argument('--detection_tokenizer', default=None, type=str, )
    parser.add_argument('--is_quote_cutoff', default=.5, type=float)
    parser.add_argument('--detection_outfile', default=None, type=str)
    parser.add_argument('--do_detection', action='store_true')
    #
    parser.add_argument('--attribution_model', type=str )
    parser.add_argument('--attribution_config', type=str, default=None)
    parser.add_argument('--attribution_tokenizer', default='gpt', type=str, )
    parser.add_argument('--attribution_outfile', type=str)
    parser.add_argument('--do_attribution', action='store_true')
    #
    parser.add_argument('--dataset_name', type=str, )
    parser.add_argument('--start_idx', default=None, type=int)
    parser.add_argument('--n_docs', default=None, type=int)
    parser.add_argument('--to_run_ids', default=None, type=str)
    parser.add_argument('--already_run_ids', default=None, type=str)

    parser.add_argument('--platform', default='local', type=str)
    args = parser.parse_args()

    from sentence_model import SentenceClassificationModel as DetectionModelClass
    from sentence_model import TokenizedDataset as DetectionDataset
    from sentence_model import collate_fn as DetectionCollateFn

    # load in dataset
    data = list(jsonlines.open(args.dataset_name))
    if args.start_idx is not None:
        data = data[args.start_idx:]

    if args.n_docs is not None:
        data = data[:args.n_docs]

    device = get_device()
    # Load the models
    args.cache_dir = None
    if args.platform == 'gcp':
        args.cache_dir = '/dev'


    if args.do_detection:
        # detection
        detection_tokenizer = AutoTokenizer.from_pretrained(args.detection_tokenizer or args.detection_model, cache_dir=args.cache_dir)
        detection_config = AutoConfig.from_pretrained(args.detection_config or args.detection_model, cache_dir=args.cache_dir)
        detection_config.classification_head = {
            'num_labels': 1,
            'pooling_method': 'average',
        }

        detection_model = DetectionModelClass.from_pretrained(args.detection_model, config=detection_config, cache_dir=args.cache_dir)
        detection_dataset = DetectionDataset(tokenizer=detection_tokenizer, do_score=True)

        # perform detection
        detection_model.eval()
        detection_model = detection_model.to(device)

        # stream the data to a file
        data_with_detection = []
        for doc in tqdm(data, total=len(data)):
            sentences = doc['sentence']
            sentences = list(map(lambda x: {'sent': x}, sentences))

            input_ids, attention_mask, _ = detection_dataset.process_one_doc(sentences)
            if input_ids is not None:
                processed_datum = {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }

                # perform quote detection score
                try:
                    score = detection_model.get_proba(**processed_datum)
                except:
                    print('detection error')
                    continue
                scores = score.cpu().detach().numpy().flatten()

                # process data
                datum_for_attribution = []
                for sent_idx, sent in enumerate(sentences):
                    output_packet = {
                        'is_quote': (float(scores[sent_idx]) > args.is_quote_cutoff),
                        'sent': sent['sent'],
                        'sent_idx': sent_idx,
                        'doc_idx': doc['doc_idx'],
                    }
                    datum_for_attribution.append(output_packet)
            data_with_detection.append(datum_for_attribution)

        if args.detection_outfile is not None:
            with open(args.detection_outfile, 'w') as f:
                jsonlines.Writer(f).write_all(data_with_detection)

    if args.do_attribution:
        fn, f_end = args.attribution_outfile.split('.')
        s, e = args.start_idx or 0, (args.start_idx or 0) + (args.n_docs or 0)
        args.attribution_outfile = fn + f'__{s}-{e}__' + '.' + f_end
        attribution_model_type, openai_key = get_attribution_model_type(args.attribution_model)
        if attribution_model_type == 'openai':
            attribution_model = OpenAIModel(args.attribution_model)
            attribution_tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
            attribution_dataset = OpenAIAttributionDataset(tokenizer=attribution_tokenizer)
            openai.api_key = KEYS[openai_key]
        else:
            from qa_model import QAModel
            from qa_dataset import QATokenizedDataset, collate_fn

            attribution_config = AutoConfig.from_pretrained(args.attribution_config or args.attribution_model, cache_dir=args.cache_dir)
            attribution_tokenizer = AutoTokenizer.from_pretrained(
                args.attribution_tokenizer or args.attribution_model, cache_dir=args.cache_dir
            )
            attribution_model = QAModel.from_pretrained(args.attribution_model, cache_dir=args.cache_dir)
            attribution_model = attribution_model.to(device)
            attribution_dataset_core = QATokenizedDataset(hf_tokenizer=attribution_tokenizer)
            attribution_dataset = QADatasetWrapper(
                qa_dataset=attribution_dataset_core, collator=collate_fn, tokenizer=attribution_tokenizer
            )

        if not args.do_detection:
            data_with_detection = list(jsonlines.open(args.detection_outfile))
            if args.already_run_ids:
                ran_ids = json.load(open(args.already_run_ids))
                data_with_detection = list(filter(lambda x: not x[0]['doc_idx'] in ran_ids, data_with_detection))
            if args.to_run_ids:
                to_run_ids = json.load(open(args.to_run_ids))
                data_with_detection = list(filter(lambda x: x[0]['doc_idx'] in to_run_ids, data_with_detection))
            if args.start_idx is not None:
                data_with_detection = data_with_detection[args.start_idx:]
            if args.n_docs is not None:
                data_with_detection = data_with_detection[:args.n_docs]

        with open(args.attribution_outfile, 'w') as f:
            writer = jsonlines.Writer(f)

            # stream the data to a file
            for datum_for_attribution in tqdm(data_with_detection, total=len(data_with_detection)):
                # perform attribution
                final_output = []
                data_for_scoring = attribution_dataset.process_one_doc(datum_for_attribution)
                for packet, datum in zip(datum_for_attribution, data_for_scoring):
                    if datum is not None and packet['is_quote']:
                        datum = attribution_dataset.prepare_sent(datum)
                        attribution = attribution_model(**datum)
                        attribution = attribution_dataset.process_output(attribution, datum)
                        packet['attribution'] = attribution
                    else:
                        packet['attribution'] = 'None'
                    final_output.append(packet)

                # stream to disk (check!!)
                writer.write(final_output)



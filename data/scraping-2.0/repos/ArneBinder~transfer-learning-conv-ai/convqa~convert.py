import bz2
import json
import logging
import os
import random
import tarfile
from collections import Counter

import numpy as np
import plac
from tqdm import tqdm

from convqa.interact import create_sentencizer

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# too large for memory
def sample_neg_indices_old(n_instances, n_candidates):
    # create index array [[0, 1, .., n_instances-1], .., [0, 1, .., n_instances-1]]
    a = np.tile(np.arange(n_instances), n_instances).reshape((n_instances, n_instances))
    # for each row, replace current idx with last
    np.fill_diagonal(a, n_instances-1)
    # truncate replaced index (last one)
    a = a[:, :-1]
    # shuffle each row
    #np.random.shuffle(a.T)
    np.apply_along_axis(np.random.shuffle, axis=1, arr=a)
    # return first n_candidates of each row
    return a[:, :n_candidates]


def sample_neg_candidates(instances, n_candidates):
    if not isinstance(instances, np.ndarray):
        instances = np.array(instances)
    #counts = Counter(instances)
    #frequencies = np.empty(len(counts), dtype=float)
    #unique = np.empty(len(counts), dtype=str)
    #indices = {}
    #for i, s in enumerate(counts):
    #    frequencies[i] = counts[s]
    #    indices[s] = i
    #    #unique[i] = s
    #u = np.array(list(counts.keys()), dtype=str)
    logger.debug('number  of instances: %i' % len(instances))
    u, c = np.unique(instances, return_counts=True)
    indices = {s: i for i, s in enumerate(u)}
    #unique = list(counts.keys())
    logger.info('collected %i unique instances' % len(u))

    n_collisions = 0
    nn_collisions = 0
    collision_log = []

    #a = np.empty(shape=(len(instances), n_candidates - 1), dtype=str)
    a = []
    for i, instance in tqdm(enumerate(instances), total=len(instances)):

        idx = indices[instance]
        current_count = c[idx]
        c[idx] = 0.0
        a.append(np.random.choice(u,  n_candidates - 1, p=c / (len(instances)-current_count)))
        c[idx] = current_count
        assert u[idx] == instance, 'mismatch'

        # much slower
        #mask = instances != instance
        #a[i] = np.random.choice(instances[mask], n_candidates - 1)

        if instance in a[i]:
            nn_collisions += 1
            collision_indices = np.nonzero(a[i] == instance)[0]
            n_collisions += len(collision_indices)
            collision_log.append('collision: %s is %i times in %s @%i' % (instance, len(collision_indices), str(a[i]), i))

    logger.info('collisions: %i (in %i instances; total: %i)' % (n_collisions, nn_collisions, len(instances)))
    for e in collision_log:
        logger.debug(e)
    return a

def count_sentences(s, sentencizer, counter=None):
    s = s.strip()
    try:
        sents = sentencizer(s)
    except Exception as e:
        if ' ' in s:
            logger.warning('could not sentencize "%s", return as ONE sentence (%s)' % (s.strip(), e))
        sents = [s]
    if counter is not None:
        counter[len(sents)] +=1
    return len(sents)

def create_instance_from_coqa(record, stats, sentencizer=None, max_sentences_qa=1, max_sentences_background=None):
    all_questions = []
    all_answers = []

    was_truncated = False
    instance = {}
    instance['background'] = record['story']
    if max_sentences_background is not None:
        if 'background' not in stats:
            stats['background'] = {'n_sents': Counter()}
        stats['background']['n_sents'][len(instance['background'])] += 1
        if len(instance['background']) > max_sentences_background:
            was_truncated = True
        instance['background'] = instance['background'][:max_sentences_background]

    assert len(record['questions']) == len(record['answers']), 'number of questions / answers mismatch'
    #instance['utterances'] = []
    instance['n_utterances'] = 0
    #history = []
    for i in range(len(record['questions'])):
        #utterance = {}
        assert record['questions'][i]['turn_id'] == record['answers'][i]['turn_id'] == i + 1, 'turn_id mismatch'
        question_text = record['questions'][i]['input_text']
        answer_text = record['answers'][i]['input_text']
        # skip answer-question pairs if number of sentences in one of them > max_sentences
        continue_this = False
        if sentencizer is not None:
            if 'question' not in stats:
                stats['question'] = {'n_sents': Counter()}
            if max_sentences_qa and count_sentences(s=question_text, sentencizer=sentencizer,
                                                    counter=stats['question']['n_sents']) > max_sentences_qa:
                continue_this = True
            if 'answer' not in stats:
                stats['answer'] = {'n_sents': Counter()}
            if max_sentences_qa and count_sentences(s=answer_text, sentencizer=sentencizer,
                                                    counter=stats['answer']['n_sents']) > max_sentences_qa:
                continue_this = True
        if continue_this:
            was_truncated = True
            continue

        all_answers.append(answer_text)
        all_questions.append(question_text)
        instance['n_utterances'] += 1

    return instance, all_questions, all_answers, was_truncated


def create_instance_from_squad(record, stats, sentencizer=None, max_sentences_qa=1, max_sentences_background=None):
    all_questions = []
    all_answers = []

    was_truncated = False
    instance = {}
    instance['background'] = record['context']
    if max_sentences_background is not None:
        if 'background' not in stats:
            stats['background'] = {'n_sents': Counter()}
        stats['background']['n_sents'][len(instance['background'])] += 1
        if len(instance['background']) > max_sentences_background:
            was_truncated = True
        instance['background'] = instance['background'][:max_sentences_background]

    instance['n_utterances'] = 0
    # shuffle because impossible questions tend to be at the end
    random.shuffle(record['qas'])
    for qa in record['qas']:
        question_text = qa['question']
        if qa['is_impossible']:
            answer_text = 'unknown'
        else:
            allowed_answers = [a['text'] for a in qa['answers']]
            answer_text = max(allowed_answers, key=len)

        # skip answer-question pairs if number of sentences in one of them > max_sentences
        continue_this = False
        if sentencizer is not None:
            if 'question' not in stats:
                stats['question'] = {'n_sents': Counter()}
            if max_sentences_qa and count_sentences(s=question_text, sentencizer=sentencizer,
                                                    counter=stats['question']['n_sents']) > max_sentences_qa:
                continue_this = True
            if 'answer' not in stats:
                stats['answer'] = {'n_sents': Counter()}
            if max_sentences_qa and count_sentences(s=answer_text, sentencizer=sentencizer,
                                                    counter=stats['answer']['n_sents']) > max_sentences_qa:
                continue_this = True
        if continue_this:
            was_truncated = True
            continue

        all_answers.append(answer_text)
        all_questions.append(question_text)
        instance['n_utterances'] += 1

    return instance, all_questions, all_answers, was_truncated



def dataset_split_to_dialog(data, instance_builder=create_instance_from_coqa, n_candidates=20,
                            create_question_utterances=False, **instance_builder_kargs
                            ):
    instances = []
    all_answers = []
    all_questions = []
    stats = {}
    n_skipped = 0
    for record in data:
        instance, current_questions, current_answers, was_truncated = instance_builder(
            record=record, stats=stats, **instance_builder_kargs)
        if was_truncated:
            n_skipped += 1
            continue
        instances.append(instance)
        all_questions.extend(current_questions)
        all_answers.extend(current_answers)
    logger.info('data created (skipped %i out of %i)' % (n_skipped, len(instances) + n_skipped))
    #logger.info('max_sentences_background: %s' % str(max_sentences_background))
    #logger.info('max_sentences_qa: %s' % str(max_sentences_qa))
    logger.info(stats)

    logger.info('sample negative answers...')
    sampled_neg_answers = sample_neg_candidates(instances=all_answers, n_candidates=n_candidates)
    sampled_neg_questions = None
    if create_question_utterances:
        logger.info('sample negative questions...')
        sampled_neg_questions = sample_neg_candidates(instances=all_questions, n_candidates=n_candidates)

    logger.info('negative samples created')
    #all_candidates = np.concatenate([sampled_neg_answers.T, [all_answers]]).T

    i = 0
    for instance in instances:
        instance['utterances'] = []
        history = []
        #for j, utterance in enumerate(instance['utterances']):
        for _ in range(instance['n_utterances']):
            if sampled_neg_questions is not None:
                new_utterance = {'history': history.copy(),
                                 'candidates': sampled_neg_questions[i].tolist() + [all_questions[i]]}
                instance['utterances'].append(new_utterance)
            history.append(all_questions[i])

            new_utterance = {'history': history.copy(),
                             'candidates': sampled_neg_answers[i].tolist() + [all_answers[i]]}
            instance['utterances'].append(new_utterance)
            history.append(all_answers[i])
            i += 1
        del instance['n_utterances']
    logger.info('candidates created')

    return instances


def convert_to_dialog(dir='/mnt/DATA/ML/data/corpora/QA/CoQA',
                      dev='coqa-dev-v1.0.json',
                      train='coqa-train-v1.0.json',
                      out=None,
                      n_candidates=20,
                      create_question_utterances=False,
                      data_loader=lambda file_name: json.load(open(file_name))['data'],
                      instance_builder=create_instance_from_coqa,
                      **instance_builder_kwargs
                      ):
    dev = os.path.join(dir, dev)
    train = os.path.join(dir, train)
    if out is None:
        dataset_name = os.path.basename(dir) or os.path.basename(os.path.dirname(dir))
        fn = '%s_converted_dialog' % dataset_name.lower()
        if instance_builder_kwargs.get('max_sentences_qa', -1) >= 0:
            fn += '_sentsqa%i' % instance_builder_kwargs['max_sentences_qa']
        if instance_builder_kwargs.get('max_sentences_background', -1) >= 0:
            fn += '_sentsb%i' % instance_builder_kwargs['max_sentences_background']
        if create_question_utterances:
            fn += '_questionutterances'

        out = os.path.join(dir, '%s.json' % fn)


    converted = {}
    logger.info('convert dev...')
    data_dev = list(data_loader(dev))
    converted['valid'] = dataset_split_to_dialog(data=data_dev, n_candidates=n_candidates, instance_builder=instance_builder,
                                                 create_question_utterances=False, **instance_builder_kwargs)
    if create_question_utterances:
        converted['valid_questionutterances'] = dataset_split_to_dialog(data=data_dev, n_candidates=n_candidates,
                                                                        instance_builder=instance_builder,
                                                                        create_question_utterances=True,
                                                                        **instance_builder_kwargs)
    logger.info('convert train...')
    data_train = data_loader(train)
    converted['train'] = dataset_split_to_dialog(data=data_train, n_candidates=n_candidates, instance_builder=instance_builder,
                                                 create_question_utterances=create_question_utterances,
                                                 **instance_builder_kwargs)

    logger.info('dump to json: %s ...' % out)
    json.dump(converted, open(out, 'w'), indent=2)
    return out


def gen_dataset_extract(fn, extract_size=10, start_idx=0):
    data = json.load(open(fn))
    if start_idx > 0:
        fn_out = fn.replace('.json', '_extract%s_start%s.json' % (str(extract_size), str(start_idx)))
    else:
        fn_out = fn.replace('.json', '_extract%s.json' % str(extract_size))

    # print dataset size
    for k in data:
        logger.info('%s: %i' % (k, len(data[k])))
    logger.info('write to: %s' % fn_out)
    if extract_size is not None:
        data = {k: data[k][start_idx:extract_size+start_idx] for k in data}
    json.dump(data, open(fn_out, 'w'), indent=2)


def convert_hotpotqa_wikidump_to_dict(fn, fields=('text', 'title')):
    entries = {}
    with tarfile.open(fn, "r:bz2") as tar:
        for tarinfo in tqdm(tar):
            f = tar.extractfile(tarinfo)
            if f is not None:
                uncomp = bz2.decompress(f.read())
                for l in uncomp.split(b'\n'):
                    if l.strip() != b'':
                        entry = json.loads(l)
                        entries[int(entry['id'])] =  {f: entry[f] for f in fields}
    return entries


def dummy_tokenize():
    from pytorch_pretrained_bert import OpenAIGPTTokenizer

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Tokenized input
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = tokenizer.tokenize(text)
    return tokenized_text


def convert_coqa(directory='/mnt/DATA/ML/data/corpora/QA/CoQA', create_question_utterances=True, max_sentences_qa=1):
    # convert CoQA to conversational QA format
    logger.info('load data from directory: %s' % directory)
    sentencizer = create_sentencizer() if max_sentences_qa >= 0 else None
    return convert_to_dialog(dir=directory,
                             dev='coqa-dev-v1.0.json',
                             train='coqa-train-v1.0.json',
                             out=None,
                             instance_builder=create_instance_from_coqa, max_sentences_qa=max_sentences_qa,
                             create_question_utterances=create_question_utterances, sentencizer=sentencizer)
    # stats: train: 7199; valid: 500



def convert_squad(directory='/mnt/DATA/ML/data/corpora/QA/SQaAD', create_question_utterances=True, max_sentences_qa=1):
    # convert SQaAD to conversational QA format
    def squad_data_loader(fn):
        data = json.load(open(fn))
        for article in data['data']:
            for paragraph in article['paragraphs']:
                yield paragraph

    sentencizer = create_sentencizer() if max_sentences_qa >= 0 else None
    logger.info('load data from directory: %s' % directory)
    return convert_to_dialog(dir=directory,
                             dev='dev-v2.0.json',
                             train='train-v2.0.json',
                             out=None,
                             data_loader=squad_data_loader,
                             instance_builder=create_instance_from_squad, max_sentences_qa=max_sentences_qa,
                             create_question_utterances=create_question_utterances, sentencizer=sentencizer)
    # stats: train: 7199; valid: 500


def main(dataset: ('the dataset', 'positional', None, str, ['CoQA', 'SQuAD']),
         *args: ('dataset specific parameters',)):
    logger.info('convert %s dataset to dialog format...' % dataset)
    if dataset == 'CoQA':
        out_fn = plac.call(convert_coqa, args)
    elif dataset == 'SQuAD':
        out_fn = plac.call(convert_squad, args)
    else:
        raise NotImplementedError('no converter for dataset "%s" implemented' % dataset)

    gen_dataset_extract(fn=out_fn, extract_size=10, start_idx=0)
    #x = dummy_tokenize()
    logger.info('done')


if __name__ == '__main__':
    plac.call(main)

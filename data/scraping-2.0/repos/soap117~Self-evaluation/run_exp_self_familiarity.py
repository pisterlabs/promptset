import os.path
import pickle
import time

from myutils import *
import types
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import metrics
import numpy as np
import csv
import openai
import re
from train_para import generate_parser
from transformers import AutoTokenizer, AutoModelForCausalLM
def get_output_scores_here(scores, sequences):
    scores_selected = []
    for i in range(len(scores)):
        score = scores[i]
        score = torch.softmax(score, dim=-1)
        #print(sequences[:, i],  score[:, sequences[:, i]])
        score = score[:, sequences[:, i]].max()
        scores_selected.append(score)
    return scores_selected

# COMMAND ----------

def back_exp(target, targets, domain, outputs_forward):
    print('________Back Reference Score__________')
    text_forward = tokenizer.decode(outputs_forward.sequences[0], skip_special_tokens=True)
    if text_forward[-5:] == 'User ':
        text_forward = text_forward[:-5]
    if text_forward[-1] == '\n':
        text_forward = text_forward[:-1]

    tokenizer.unk_token = args.unk_token
    LEADING_DECODING_BACK = "\"{contents}\" is related to what?"
    question_backward = LEADING_DECODING_BACK.format(contents=text_forward)
    question_backward = PROMPT_FOR_GENERATION_FORMAT.format(instruction=question_backward)
    question_backward_masked = mask_input_words(question_backward, targets, tokenizer)
    input_ids_mask = tokenizer(question_backward_masked, return_tensors="pt").input_ids.to(my_device)
    ori_target = targets[-1]
    target_words = list(set([ori_target, ori_target.lower(), ori_target.upper(), ' '.join([x.capitalize() for x in ori_target.split()])]))
    print(target_words)
    target_words = target_words + [' ' + x for x in target_words]
    force_words_ids = tokenizer(target_words, add_special_tokens=False).input_ids

    try:
        outputs_constraint = model.generate(
            input_ids_mask,
            force_words_ids=[force_words_ids],
            output_scores=True,
            return_dict_in_generate=True,
            max_length=input_ids_mask.shape[-1]+len(force_words_ids)+args.backward_search_length,
            stopping_criteria=stopping_criteria,
            num_beams=args.backward_search_size,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        backward_text = tokenizer.decode(outputs_constraint.sequences[0])
        print(backward_text)
        back_score = torch.exp(outputs_constraint.sequences_scores).item()
    except:
        back_score = 0
        backward_text = ""

    return back_score, backward_text

def forward_exp(target, targets, domain):
    tokenizer.unk_token = args.unk_token
    question = question_template.format(concept=target)
    question = PROMPT_FOR_GENERATION_FORMAT.format(instruction=question)
    outputs_forward, original_decoding_scores = ask_original_question(question, model, tokenizer,
                                                              stopping_criteria=stopping_criteria)
    return 0, outputs_forward

def percentile_5(x):
    return np.percentile(x, 5)

def benchmark_exp(data_common_file):
    with open(data_common_file, newline='', encoding='utf-8') as csvfile:
        if os.path.exists('./data/{mpt}-{domain}-benchmark.pkl'.format(mpt=mpt, domain=args.exp_file)):
            intervals = pickle.load(open('./data/{mpt}-{domain}-benchmark.pkl'.format(mpt=mpt, domain=args.exp_file), 'rb'))
            f_intervals = intervals['f_intervals']
            b_intervals = intervals['b_intervals']
        else:
            fscores = []
            bscores = []
            reader = csv.DictReader(csvfile)
            c = 0
            for row in reader:
                target = row['Concept']
                domain = row['\ufeffDomain'].lower()
                targets = word_tokenize(target)
                # remove the stop words in targets
                targets = [x for x in targets if x.lower() not in stop_words] + [target]
                print('+++++++++++++++++++')
                print(target, targets)
                fscore, outputs_forward = forward_exp(target, targets, domain)
                bscore, _ = back_exp(target, targets, domain, outputs_forward)
                print(row)
                print(fscore, bscore)
                fscores.append(fscore)
                bscores.append(bscore)
                print('-------------------')
                c += 1
            from scipy.stats import bootstrap
            rng = np.random.default_rng()
            fscores = np.array(fscores)
            bscores = np.array(bscores)
            res_fscore = bootstrap((fscores,), percentile_5, confidence_level=0.95,
                            random_state=rng)
            res_bscore = bootstrap((bscores,), percentile_5, confidence_level=0.95,
                            random_state=rng)
            f_intervals = res_fscore.confidence_interval
            b_intervals = res_bscore.confidence_interval
            print(f_intervals, b_intervals)
            print(fscores)
            print(bscores)
            intervals = {'f_intervals': f_intervals, 'b_intervals': b_intervals, 'fscores': fscores, 'bscores': bscores}
            pickle.dump(intervals, open('./data/{mpt}-{domain}-benchmark.pkl'.format(mpt=mpt, domain=args.exp_file), 'wb'))
        return f_intervals, b_intervals

def labeling(data_file):
    template = \
'''
Using the background information provided, assess the participant's understanding of the target concept from their explanation. Please provide a rating between 1 and 9, with 9 being the best score (9 = Excellent, 5 = Average, 1 = Poor).

Concept:
{concept}

Background:
{background}

Participant's Explanation:
{response}

Scoring Guide:
Award a score of "7-9 (Excellent)" if the participant's explanation of the concept is correct, with the inclusion of essential details demonstrating a comprehensive understanding of the concept. Minor inaccuracies can be ignored.
Award a score of "4-6 (Average)" if the participant's explanation of the concept is partially correct but lacks essential details.
Award a score of "1-3 (Poor)" if the participant's explanation of the concept is incorrect or includes obvious errors.

The term "essential details" refers to those details that only someone knowledgeable in this concept would suggest or that capture the core idea of the concept. These details cannot be deduced from a simple literal understanding.
Please note that the length of the explanation is not a determinant of the score. A concise yet accurate explanation with essential details is worthy of a high score.
Your Score (please provide a number between 1 and 9):
'''
    with open(data_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        labels = []
        messages = []
        if os.path.exists('./data/{mpt}-{domain}-messages.pkl'.format(mpt=mpt, domain=args.exp_file)):
            messages = pickle.load(open('./data/{mpt}-{domain}-messages.pkl'.format(mpt=mpt, domain=args.exp_file), 'rb'))
            for message_one in messages:
                labels.append(message_one[4] if message_one[4] is not None else True)
        else:
            for row in reader:
                real = row['Real']
                if real == 'TRUE':
                    target = row['Concept']
                    domain = row['\ufeffDomain'].lower()
                    question = question_template_long.format(
                        concept=target,
                        domain=domain)
                    question = PROMPT_FOR_GENERATION_FORMAT.format(instruction=question)
                    outputs_forward, _ = ask_original_question(question, model, tokenizer, stopping_criteria,
                                                               max_length=args.forward_search_length*2)
                    response = tokenizer.decode(outputs_forward.sequences[0], skip_special_tokens=True)
                    if response[-5:] == 'User ':
                        response = response[:-5]
                    if response[-1] == '.':
                        response = response[:-1]
                    background = row['Background']
                    # removd [.*] from background
                    background = re.sub(r'\[\^.\^\]', '', background)
                    question = template.format(background=background, concept=row['Concept'], response=response)
                    while True:
                        try:
                            # if the time is too long, the connection will be broken
                            response = openai.ChatCompletion.create(
                                model="gpt-4-0613",
                                temperature=0,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": question},
                                ],
                                request_timeout = 30,
                            )
                            break
                        except Exception as e:
                            print(e)
                            time.sleep(5)
                            continue
                    print(question)
                    message = response['choices'][0]['message']['content']
                    print(message)
                    # keep only the digits and .
                    message_number = re.sub(r'[^\d.]', '', message)
                    score = float(message_number)
                    messages.append((target, question, message, score))
                    if score < 5 and score >= 1:
                        label = False
                    elif score > 5 and score <= 9:
                        label = True
                    else:
                        # decide by manual
                        while True:
                            manual = input(
                                'Please judge if the response explains the concept correctly instead of guessing by literal meaning. Please answer in [ Good , Bad ].')
                            if manual in ['Good', 'Bad']:
                                break
                        if manual == 'Good':
                            label = True
                        else:
                            label = False

                    print(label)
                else:
                    label = False
                labels.append(label)
            pickle.dump(messages, open('./data/{mpt}-{domain}-messages.pkl'.format(mpt=mpt, domain=args.exp_file), 'wb'))
            pickle.dump(labels, open('./data/{mpt}-{domain}-labels.pkl'.format(mpt=mpt, domain=args.exp_file), 'wb'))
        return labels

def cal_score_min(bscores, b_intervals):
    pred = np.array([min(x[0]) for x in bscores])
    y = np.array([int(x[1]) for x in bscores])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    accuracy_bscore = metrics.accuracy_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    f1_bscore = metrics.f1_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    auc_bscore = metrics.auc(fpr, tpr)
    return accuracy_bscore, f1_bscore, auc_bscore

def sum_scores(scores):
    ans = 0
    ratio = 0
    t = 1.0
    for score in scores:
        ans += t*score
        ratio += t
        t/=2
    ans = ans/ratio
    return ans

def cal_score_accumulate(bscores, b_intervals):
    pred = np.array([sum_scores(x[0]) for x in bscores])
    y = np.array([int(x[1]) for x in bscores])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    accuracy_bscore = metrics.accuracy_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    f1_bscore = metrics.f1_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    auc_bscore = metrics.auc(fpr, tpr)
    return accuracy_bscore, f1_bscore, auc_bscore


def cal_score_first(bscores, b_intervals):
    pred = np.array([x[0][0] for x in bscores])
    y = np.array([int(x[1]) for x in bscores])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    accuracy_bscore = metrics.accuracy_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    f1_bscore = metrics.f1_score(y, pred > (b_intervals[0] + b_intervals[1]) / 2)
    auc_bscore = metrics.auc(fpr, tpr)
    return accuracy_bscore, f1_bscore, auc_bscore

from itertools import combinations

def merge_entities(entities, question):
    # initialize an empty list to store the final entities
    final_entities = []
    # iterate over every combination of 2 entities
    for entity1, entity2 in combinations(entities, 2):
        # create the merged entities
        merged_no_space = entity1 + entity2
        merged_with_space = entity1 + ' ' + entity2

        # if the merged entity is in the question, add it to the final entities
        if merged_no_space.lower() in question.lower() or merged_with_space.lower() in question.lower():
            final_entities.append(merged_no_space if merged_no_space in question else merged_with_space)
        else:
            # if no match found, add the original entities
            if entity1 not in ' '.join(final_entities):
                final_entities.append(entity1)
            if entity2 not in ' '.join(final_entities):
                final_entities.append(entity2)
    for entity in entities:
        if entity not in ' '.join(final_entities):
            final_entities.append(entity)
    if len(final_entities) == len(entities):
        return final_entities
    else:
        return merge_entities(final_entities, question)
def remove_covered_entities(entities):
    final_entities = []
    entities_sorted = sorted(entities, key=lambda x: len(x), reverse=True)
    existing = ''
    for entity in entities_sorted:
        if entity in existing:
            continue
        else:
            existing += entity
            final_entities.append(entity)
    return final_entities



def start_test(data_file, data_common_file):
    fscores = []
    bscores = []
    with torch.no_grad():
        f_intervals, b_intervals = benchmark_exp(data_common_file)
        labels = labeling(data_file)
        with open(data_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            c = 0
            for row, label in zip(reader, labels):
                concept = row['Concept']
                extracted_concepts = eval(row['Entities'])
                questions = eval(row['Questions'])
                domain = row['\ufeffDomain'].lower()
                print('+++++++++++++++++++')
                for extracted_concept, question in zip(extracted_concepts, questions):
                    print("original: ", extracted_concept)
                    extracted_concept = sorted(extracted_concept, key=lambda x: question.lower().find(x.lower()))
                    extracted_concept = merge_entities(extracted_concept, question)
                    extracted_concept_cool = remove_covered_entities(extracted_concept)
                    extracted_concept = [x for x in extracted_concept if x in extracted_concept_cool]
                    new_concept = []
                    for concept_one in extracted_concept:
                        if concept_one not in common_english_words:
                            new_concept.append(concept_one)
                    if len(new_concept) != 0:
                        extracted_concept = new_concept
                    # sort by their position in question
                    extracted_concept = sorted(extracted_concept, key=lambda x: entity_rare(x), reverse=False)
                    print("after: ", extracted_concept)
                    fscore_group = []
                    bscore_group = []
                    text_forwards = []
                    text_backwards = []
                    for target in extracted_concept:
                        targets = word_tokenize(target)
                        # remove the stop words in targets
                        targets = [x for x in targets if x.lower() not in stop_words] + [target]
                        print(target, targets)
                        fscore, output_forward = forward_exp(target, targets, domain)
                        text_forward = tokenizer.decode(output_forward.sequences[0], skip_special_tokens=True)
                        bscore, text_backward = back_exp(target, targets, domain, output_forward)
                        fscore_group.append(fscore)
                        bscore_group.append(bscore)
                        text_forwards.append(text_forward)
                        text_backwards.append(text_backward)
                        print(target, bscore)
                    print('<<<<<<<<<<<<<<<<<<')
                    print(concept, bscore_group)
                    print('<<<<<<<<<<<<<<<<<<')
                    if len(fscore_group) == 0:
                        fscore_group = [0]
                    if len(bscore_group) == 0:
                        bscore_group = [0]
                    fscores.append((fscore_group, label, extracted_concept, text_forwards))
                    bscores.append((bscore_group, label, extracted_concept, text_backwards))
                c += 1
                # calculate the auc and accuracy
            fmetrics = {}
            bmetrics = {}
            for metric in [cal_score_min, cal_score_accumulate, cal_score_first]:
                print(metric.__name__)
                accuracy_bscore, f1_bscore, auc_bscore = metric(bscores, b_intervals)
                bmetrics[metric.__name__] = [auc_bscore, accuracy_bscore, f1_bscore]
                print(auc_bscore, accuracy_bscore, f1_bscore)
                accuracy_fscore, f1_ascore, auc_fscore = metric(fscores, f_intervals)
                fmetrics[metric.__name__] = [auc_fscore, accuracy_fscore, f1_ascore]
                print(auc_fscore, accuracy_fscore, f1_ascore)
        exp_results = {'fscores': fscores, 'bscores': bscores,
                       'fmetrics': fmetrics, 'bmetrics': bmetrics,
                       'f_intervals': f_intervals, 'b_intervals': b_intervals, 'mpt': mpt}
        pickle.dump(exp_results,
                    open('./data/{mpt}-{domain}-general-exp_results.pkl'.format(mpt=mpt, domain=args.exp_file),
                         'wb'))
        print(fscores)
        print(bscores)

def entity_rare(entity):
    entity = entity.split(' ')
    rare = 1
    for word in entity:
        if word not in word_rare or word[0].isupper():
            rare *= np.exp(-len(word_rare) / 100)
        else:
            rare*=word_rare[word]
    return rare

if __name__ == "__main__":
    para_list = [
                    ['lmsys/vicuna-13b-v1.3', '...', 856, ['</s>'], 200, 15],
    ]
    common_english_words_file = './data/wiki-100k.txt'
    common_english_words = []
    with open(common_english_words_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            if line[0] == '#':
                continue
            common_english_words.append(line.strip())
    word_rare = {}
    for id, word in enumerate(common_english_words):
        word_rare[word] = np.exp(-(id + 1) / 100)
    common_english_words = common_english_words[0:10000]

    parser = generate_parser()
    args = parser.parse_args()
    data_file = 'data/data_knowledge_{}_general_multi.csv'.format(args.exp_file)
    data_common_file = 'data/data_common.csv'
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = ""
    PROMPT_FOR_GENERATION_FORMAT_M = \
"""
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)
    PROMPT_FOR_GENERATION_FORMAT_V = "USER: {instruction} ASSISTANT:"
    PROMPT_FOR_GENERATION_FORMAT_F = "User: {instruction}\nAssistant:"
    for para in para_list:
        args.model = para[0]
        args.stop_token = para[3]
        args.unk_token = para[1]
        args.unk_token_id = para[2]
        args.forward_search_length = para[4]
        args.backward_search_length = para[5]
        question_template = "Explain the \"{concept}\" within one short paragraph."
        question_template_long = "Explain the \"{concept}\" within one paragraph with details."
        if 'vicuna' in args.model:
            PROMPT_FOR_GENERATION_FORMAT = PROMPT_FOR_GENERATION_FORMAT_V
        elif 'mpt' in args.model or 'dolly' in args.model or 'alpaca' in args.model:
            PROMPT_FOR_GENERATION_FORMAT = PROMPT_FOR_GENERATION_FORMAT_M
        elif 'falcon' in args.model:
            PROMPT_FOR_GENERATION_FORMAT = PROMPT_FOR_GENERATION_FORMAT_F

        print(args)
        max_memory = {0: "50GiB", 1: "40GiB", 2: "40GiB", "cpu": "40GiB"}
        mpt = args.model.split('/')[-1]
        model = AutoModelForCausalLM.from_pretrained("{mpt}".format(mpt=args.model), cache_dir='/data/cache',
                                                     trust_remote_code=True, device_map="sequential",
                                                     max_memory=max_memory, torch_dtype=torch.bfloat16)
        model.eval()
        if 'alpaca' in args.model:
            tokenizer = AutoTokenizer.from_pretrained("{mpt}".format(mpt=args.model), cache_dir='/data/cache',
                                                      unk_token="<unk>",
                                                      bos_token="<s>",
                                                      eos_token="</s>")
        else:
            tokenizer = AutoTokenizer.from_pretrained("{mpt}".format(mpt=args.model), cache_dir='/data/cache')
        if args.unk_token_id is None:
            args.unk_token_id = tokenizer.vocab[args.unk_token]
        tokenizer.unk_token_id = args.unk_token_id
        tokenizer.unk_token = args.unk_token
        print('Load Model')
        # id2word = {v:k for k, v in tokenizer.get_vocab().items()}
        # tokenizer.id2word = id2word
        # tokenizer.convert_ids_to_tokens = types.MethodType(convert_ids_to_tokens, tokenizer)

        # model.to(torch.device('cpu'))

        # COMMAND ----------
        stop_words = args.stop_token
        stop_words_ids = [
            tokenizer.vocab[stop_word] for stop_word in stop_words]
        s1 = StoppingCriteriaSub(stops=stop_words_ids, encounters=1)
        print(stop_words_ids)
        stopping_criteria = StoppingCriteriaList([s1])
        start_test(data_file, data_common_file)
        del model
        torch.cuda.empty_cache()
        # COMMAND ----------
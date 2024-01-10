# 各个算法比较

# pip install git+https://github.com/Liadrinz/transformers-unilm
# pip install transformers
# 安装torch
# pip install nltk

import argparse
import csv
import datetime
import json
import random
import time
from functools import partial

import data_convertor
import detect_gpt
import gltr
import hc3_ling
import hc3_single
import llmdet
import openai_roberta_base
import openai_roberta_large
import radar_vicuna

support_methods = [
    'gltr',
    'hc3_ling',
    'hc3_single',
    'intrinsic-dim',
    'llmdet',
    'openai-roberta-base',
    'openai-roberta-large',
    'radar-vicuna',
    'detect_gpt',
]

support_datasets = [
    'CHEAT',
    'ghostbuster',
    'hc3_english',
    'hc3_plus_english',
    'm4',
    'arxiv_cs'
]


def get_classifier(method):
    start_time = time.time()

    classifier = None

    if method == 'detect_gpt':
        model = detect_gpt.init_model()

        def classify(text):
            return detect_gpt.classify_is_human(model, text=text)

        classifier = classify

    if method == 'gltr':
        model = gltr.LM()

        def classify(text):
            return gltr.classify_is_human(model, text=text, token_bar=0.6500)

        classifier = classify

    if method == 'hc3_ling':
        model = None

        def classify(text):
            return hc3_ling.classify_is_human(text=text)

        classifier = classify

    if method == 'hc3_single':
        model = hc3_single.init_classifier()

        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        classifier = classify

    if method == 'intrinsic-dim':
        model = hc3_single.init_classifier()

        def classify(text):
            return hc3_single.classify_is_human(model, text=text)

        classifier = classify

    if method == 'llmdet':
        model = llmdet.init_models()

        def classify(text):
            return llmdet.classify_is_human(model, text=text)

        classifier = classify

    if method == 'openai-roberta-base':
        model = openai_roberta_base.init_classifier()

        def classify(text):
            return openai_roberta_base.classify_is_human(model, text=text)

        classifier = classify

    if method == 'openai-roberta-large':
        model = openai_roberta_large.init_classifier()

        def classify(text):
            return openai_roberta_large.classify_is_human(model, text=text)

        classifier = classify

    if method == 'radar-vicuna':
        model = radar_vicuna.init_classifier()

        def classify(text):
            return radar_vicuna.classify_is_human(model, text=text)

        classifier = classify

    end_time = time.time()
    print("time to init model and classifier was {} seconds.".format(end_time - start_time))

    if classifier is None:
        print("None Method")
        return None

    def out_classifier(text: str, retry_times=3):
        if retry_times == 0:
            try:
                return classifier(text[0: 500])
            except Exception as e:
                print(e)
                print(text)
                return True
        try:
            return classifier(text)
        except Exception as e:
            print(e)
            print(text)
            return out_classifier(text[0: int(len(text) * 3 / 4)], retry_times - 1)

    return out_classifier


def get_test_data(test_dataset, test_dataset_path, test_data_nums, shuffle=False, test_file_name=None, test_key=None):
    start_time = time.time()
    result = {
        'human': [],
        'ai': []
    }

    tmp_result = []

    if test_dataset == 'CHEAT':
        tmp_result = data_convertor.convert_CHEAT_dataset(test_dataset_path)
    if test_dataset == 'ghostbuster':
        tmp_result = data_convertor.convert_ghostbuster_dataset(test_dataset_path)
    if test_dataset == 'hc3_english':
        tmp_result = data_convertor.convert_hc3_english(test_dataset_path)
    if test_dataset == 'hc3_plus_english':
        tmp_result = data_convertor.convert_hc3_plus_english(test_dataset_path)
    if test_dataset == 'm4':
        tmp_result = data_convertor.convert_m4(test_dataset_path)
    if test_dataset == 'arxiv_cs':
        tmp_result = data_convertor.convert_arxiv_cs(test_dataset_path, test_file_name, test_key)

    result['human'] = [x for x in tmp_result if x['label'] == 0]
    result['ai'] = [x for x in tmp_result if x['label'] == 1]

    # 打乱数据
    if shuffle:
        random.shuffle(result['human'])
        random.shuffle(result['ai'])

    result['human'] = result['human'][0: min(test_data_nums, len(result['human']))]
    result['ai'] = result['ai'][0: min(test_data_nums, len(result['ai']))]

    # 截断过长的数据
    for i in range(0, len(result['human'])):
        words = result['human'][i]['content'].split(' ')
        if len(words) > 500:
            result['human'][i]['content'] = " ".join(words[0: 500])
    for i in range(0, len(result['ai'])):
        words = result['ai'][i]['content'].split(' ')
        if len(words) > 500:
            result['ai'][i]['content'] = " ".join(words[0: 500])

    end_time = time.time()
    print("time to load test dataset was {} seconds.".format(end_time - start_time))

    if len(result['human']) == 0 and len(result['ai']) == 0:
        raise ValueError("load test data error: no data, please check the path:" + test_dataset_path)

    return result


def simple_test(method, test_dataset, test_dataset_path, test_data_nums):
    classifier = get_classifier(method)
    data_set = get_test_data(test_dataset, test_dataset_path, test_data_nums)
    test_result = test_classifier_and_dataset(classifier, data_set)
    test_result['method'] = method
    test_result['dataset'] = test_dataset
    test_result['dataset_path'] = test_dataset_path
    return test_result


def test_classifier_and_dataset(classifier, data_set):
    start_time = time.time()

    human_true = 0
    human_total = 0
    human_true_rate = 0.0
    ai_true = 0
    ai_total = 0
    ai_true_rate = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0

    all_data = data_set['human'] + data_set['ai']
    i = 0
    for data in all_data:
        i += 1
        content = data['content']
        label = data['label']
        pred_label = classifier(content)
        if label == 0:
            try:
                if pred_label:
                    human_true += 1
                human_total += 1
            except Exception as e:
                print(e)
                print(content)
        elif label == 1:
            try:
                if not pred_label:
                    ai_true += 1
                ai_total += 1
            except Exception as e:
                print(e)
                print(content)
        percent = round(1.0 * (i) / len(all_data) * 100, 2)
        print('test process : %s [%d/%d]' % (str(percent) + '%', i, len(all_data)), end='\r')
    print("test process end", end='\n')

    if human_total != 0:
        human_true_rate = human_true / human_total
    if ai_total != 0:
        ai_true_rate = ai_true / ai_total

    if ai_total != 0 and human_total != 0:
        if (ai_true + (human_total - human_true)) != 0:
            precision = ai_true / (ai_true + (human_total - human_true))
        recall = ai_true / ai_total
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)

    end_time = time.time()
    print("time to test {} seconds.".format(end_time - start_time))

    test_result = {
        "human_true": human_true,
        "human_total": human_total,
        "human_true_rate": human_true_rate,
        "ai_true": ai_true,
        "ai_total": ai_total,
        "ai_true_rate": ai_true_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "run_seconds": end_time - start_time
    }

    print(test_result)

    return test_result


def multi_test(method, test_datasets, test_dataset_paths, test_data_nums, test_file_names=None, test_keys=None):
    print("method test begin:" + method)
    start_time = datetime.datetime.now()
    classifier = get_classifier(method)
    data_sets = []
    multi_test_result = []
    if test_file_names == None:
        test_file_names = [None for i in range(0, min(len(test_datasets), len(test_dataset_paths)))]
    if test_keys == None:
        test_keys = [None for i in range(0, min(len(test_datasets), len(test_dataset_paths)))]
    for i in range(0, min(len(test_datasets), len(test_dataset_paths))):
        print("begin test dataset: " + test_datasets[i])
        data_set = get_test_data(test_datasets[i], test_dataset_paths[i], test_data_nums, test_file_name=test_file_names[i], test_key=test_keys[i])
        data_sets.append(data_set)
        multi_test_result.append(test_classifier_and_dataset(classifier, data_set))
        multi_test_result[i]['dataset'] = test_datasets[i]
        multi_test_result[i]['method'] = method
        multi_test_result[i]['dataset_path'] = test_dataset_paths[i]
        print("end test dataset: " + test_datasets[i])
        print(f"finished: {i + 1},  remained: {min(len(test_datasets), len(test_dataset_paths)) - i}")
    # for i in range(0, min(len(test_datasets), len(test_dataset_paths))):
    #     multi_test_result[i]['dataset'] = test_datasets[i]
    #     multi_test_result[i]['method'] = method
    #     multi_test_result[i]['dataset_path'] = test_dataset_paths[i]
    print("method test end:" + method)
    end_time = datetime.datetime.now()
    print(str(method) + " time to test {} seconds.".format(end_time - start_time))
    return multi_test_result


def test_hc3(method, direct_files):
    print("method test begin:" + method)
    start_time = datetime.datetime.now()
    classifier = get_classifier(method)
    test_results = []
    for direct_file in direct_files:
        print(direct_file)
        tmp_test_datas = []
        test_datas = {
            'human': [],
            'ai': []
        }
        with open(direct_file, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                i+=1
                if i > 200:
                    break
                json_obj = json.loads(line)
                tmp_test_datas.append({
                    "label":0,
                    "content": json_obj['human_answers'][0]
                })
                tmp_test_datas.append({
                    "label": 1,
                    "content": json_obj['chatgpt_answers'][0]
                })
        test_datas['human'] = [x for x in tmp_test_datas if x['label'] == 0]
        test_datas['ai'] = [x for x in tmp_test_datas if x['label'] == 1]
        # 截断过长的数据
        for i in range(0, len(test_datas['human'])):
            words = test_datas['human'][i]['content'].split(' ')
            if len(words) > 500:
                test_datas['human'][i]['content'] = " ".join(words[0: 500])
        for i in range(0, len(test_datas['ai'])):
            words = test_datas['ai'][i]['content'].split(' ')
            if len(words) > 500:
                test_datas['ai'][i]['content'] = " ".join(words[0: 500])

        test_result = test_classifier_and_dataset(classifier, test_datas)
        print(test_result)
        test_result['method'] = method
        test_result['file'] = direct_file
        test_results.append(test_result)

    end_time = datetime.datetime.now()
    print("end: " + str(end_time - start_time))
    return test_results


def test_moe_file(method, direct_files):
    print("method test begin:" + method)
    start_time = datetime.datetime.now()
    classifier = get_classifier(method)
    test_results = []
    for direct_file in direct_files:
        print(direct_file)
        test_datas = {
            'human': [],
            'ai': []
        }
        with open(direct_file, 'r', encoding='utf-8') as f:
            json_arr = json.load(f)
        random.shuffle(json_arr)
        test_datas['human'] = [x for x in json_arr if x['label'] == 0][0:200]
        test_datas['ai'] = [x for x in json_arr if x['label'] == 1][0:200]
        # 截断过长的数据
        for i in range(0, len(test_datas['human'])):
            words = test_datas['human'][i]['content'].split(' ')
            if len(words) > 500:
                test_datas['human'][i]['content'] = " ".join(words[0: 500])
        for i in range(0, len(test_datas['ai'])):
            words = test_datas['ai'][i]['content'].split(' ')
            if len(words) > 500:
                test_datas['ai'][i]['content'] = " ".join(words[0: 500])

        test_result = test_classifier_and_dataset(classifier, test_datas)
        print(test_result)
        test_result['method'] = method
        test_result['file'] = direct_file
        test_results.append(test_result)

    end_time = datetime.datetime.now()
    print("end: " + str(end_time - start_time))
    return test_results



def test_classifier_and_datasets(classifier, data_sets):
    result = []
    for data_set in data_sets:
        result.append(test_classifier_and_dataset(classifier, data_set))
    return result


def output_test_result_table(results, output_file_name=None):
    if output_file_name is None:
        output_file_name = 'output_result' + str(datetime.datetime.now()) + '.csv'
    if isinstance(results, list):
        pass
    else:
        results = [results]
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        # 写入标题行
        writer.writeheader()
        # 写入数据行
        writer.writerows(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gltr', help='baseline method,')
    parser.add_argument('--test_dataset', type=str, default='hc3_english', help='test dataset')
    parser.add_argument('--test_dataset_path', type=str, default='../data_collector/test_data/hc3_english',
                        help='test dataset path')
    parser.add_argument('--test_data_nums', type=int, default=1000)

    args = parser.parse_args()

    if args.method not in support_methods:
        raise ValueError('method not supported')
    for td in args.test_dataset.split(','):
        if td not in support_datasets:
            raise ValueError('test dataset not supported:' + td)
    if args.test_data_nums <= 0:
        raise ValueError('test nums must > 0')

    # test load classifier
    # classify = get_classifier(args.method)
    # sentence = "DetectGPT is an amazing method to determine whether a piece of text is written by large language models (like ChatGPT, GPT3, GPT2, BLOOM etc). However, we couldn't find any open-source implementation of it. Therefore this is the implementation of the paper."
    # print(classify(sentence))

    # test load data set
    # test_data_set = get_test_data('CHEAT', '../data_collector/test_data/CHEAT', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('m4', '../data_collector/test_data/m4', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('ghostbuster', '../data_collector/test_data/ghostbuster', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('hc3_english', '../data_collector/test_data/hc3_english', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))
    #
    # test_data_set = get_test_data('hc3_plus_english', '../data_collector/test_data/hc3_plus_english', args.test_data_nums)
    # print(len(test_data_set['human']))
    # print(len(test_data_set['ai']))

    # test simple test
    # print(simple_test(args.method, args.test_dataset, args.test_dataset_path, args.test_data_nums))

    # test multi test
    # output_test_result_table(multi_test(args.method, args.test_dataset.split(','), args.test_dataset_path.split(','), args.test_data_nums), output_file_name=args.method + '_output_result' + str(datetime.datetime.now()) + '.csv')
    # python3 benchmark.py --test_data_nums 1000 --method hc3_single --test_dataset CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english --test_dataset_path ../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english
    # python3 benchmark.py --test_data_nums 1000 --method hc3_single --test_dataset CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english --test_dataset_path ../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english

    # for method in support_methods:
    #     output_test_result_table(multi_test(method, 'CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english'.split(','),
    #                                         '../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english'.split(
    #                                             ','), 10))


    # output_test_result_table(multi_test('gltr', 'CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english'.split(','),
    #                                     '../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english'.split(
    #                                         ','), 100))

    # output_test_result_table(multi_test('detect_gpt', 'CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english'.split(','),
    #                                     '../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english'.split(
    #                                         ','), 1000))

    # output_test_result_table(multi_test('hc3_ling', 'CHEAT,m4,ghostbuster,hc3_english,hc3_plus_english'.split(','),
    #                                     '../data_collector/test_data/CHEAT,../data_collector/test_data/m4,../data_collector/test_data/ghostbuster,../data_collector/test_data/hc3_english,../data_collector/test_data/hc3_plus_english'.split(
    #                                         ','), 1000))

    # for key in ['rewrite', 'replace', 'continue', 'academic', 'summarize']:
    #     print(key + ' begin')
    #     result = []
    #     for method in support_methods:
    #         result.append(multi_test(method, 'arxiv_cs'.split(','),
    #                                             '../data_collector/test_data/arxiv_cs'.split(
    #                                                 ','), 1000, [key + '_1.jsonl'], [key])[0])
    #     output_test_result_table(result)

    # for method in support_methods:
    #     output_test_result_table(test_hc3(method,
    #                                       [
    #                                           '../data_collector/test_data/hc3_english/finance.jsonl',
    #                                           '../data_collector/test_data/hc3_english/medicine.jsonl',
    #                                           '../data_collector/test_data/hc3_english/open_qa.jsonl',
    #                                           '../data_collector/test_data/hc3_english/wiki_csai.jsonl',
    #                                        ]
    #                                       ))

    moe_lables = [
    "medicine",
    "law",
    "computer science",
    "finance",
    "pedagogy",
    "biology",
    "psychology",
    "political",
    "sports",
    "chemistry"
    ]
    moe_files = []
    for label in moe_lables:
        moe_files.append('../my_detector/adversarial_test/tmp/train_1/' + label + '.test')
    for method in support_methods:
        output_test_result_table(test_moe_file(method, moe_files))


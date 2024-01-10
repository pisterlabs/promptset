import argparse
import os
import pathlib
import itertools
import random
import sys
import time
from math import factorial

import nltk

import torch
from byt5 import byt5trainer

from JoinEval import JoinEval

MATCHING_TYPE = 'edit_dist'  # [edit_dist, exact]
NUMBER_OF_EXAMPLES_FOR_JOIN = 5

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.absolute())
DS_PATH = BASE_PATH + "/data/Datasets/FF_AJ_Splitted/"
# DS_PATH = BASE_PATH + "/data/Datasets/DXF_Splitted/"
# DS_PATH = BASE_PATH + "/data/Datasets/Synthetic_basic_20_50_Splitted/"
# DS_PATH = BASE_PATH + "/data/Datasets/Single_Reverse_10_50_Splitted/"
# DS_PATH = BASE_PATH + "/data/Datasets/Single_Substr_10_50_Splitted/"

DS_NAME = pathlib.PurePath(DS_PATH).name

OUT_FILE_PATH = BASE_PATH + f"/data/output_test/join_{DS_NAME}_{NUMBER_OF_EXAMPLES_FOR_JOIN:02}samp_{MATCHING_TYPE}.csv"


sys.path.append(str(pathlib.Path(__file__).absolute().parent))
import utils

MODEL_NAME = "google/byt5-base"
# MODEL_NAME = "gpt3"
# MODEL_NAME = "google/byt5-base,gpt3"

SAMPLES_PER_EACH_INSTANCE = 2
FAST_SAMPLING = False

NO_FRAMEWORK = False

MODEL_PATH = BASE_PATH + "/models/byt5-base-basic_synth_02000_10-checkpoints/best-checkpoint.ckpt"


transform = None
MODEL = None
TOKENIZER = None
SRC_PREPARE = utils.src_prepare
transform2 = None

OPENAI = None

USE_GPU = True

device = None

GPT3_PRINT = True


def byt5_transform(value):
    tokenizer = TOKENIZER
    model = MODEL
    inps = byt5trainer.TransformationDataset.get_src_encoding(value, tokenizer).to(device)
    gen_ids = model.generate(
        input_ids=inps['input_ids'],
        attention_mask=inps['attention_mask'],
        num_beams=1,
        max_length=50,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        # use_cache=True
    )

    preds = []
    for gen_id in gen_ids:
        try:
            dec_res = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except ValueError:
            # dec_res = ''.join(['?' for idd in gen_id])
            unk = tokenizer.encode("?")[0]
            new_id = [t if 0 <= t <= 255 else torch.tensor(unk, device=device) for t in gen_id]
            dec_res = tokenizer.decode(new_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        preds.append(dec_res)

    # preds = [
    #     tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     for gen_id in gen_ids
    # ]

    assert len(preds) == len(gen_ids)
    return "".join(preds)


def gpt3_transformer(value):
    openai = OPENAI
    rem = True
    while rem:
        try:
            response = openai.Completion.create(
                engine="text-curie-001",
                prompt=value,
                temperature=0.7,
                max_tokens=20,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            rem = False
        except openai.error.RateLimitError:
            print("Waiting 10 Sec...")
            time.sleep(10)

    res = response.choices[0].text.split("</eoe>")[0]
    # print(response)
    if GPT3_PRINT:
        print("      inp: ", value, 'out:', res)
    time.sleep(0.03)  # to deal with API limitations

    return res


def get_pairs_from_files(ds_path, tbl_names=[]):
    assert os.path.isdir(ds_path)
    dirs = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]

    res = {}

    for dir in dirs:

        typ = dir[:5]
        dir_name = dir[5:]
        assert typ in ('test_', 'smpl_')
        typ = typ[:-1]

        if len(tbl_names) > 0 and dir_name not in tbl_names:
            continue

        ds_dir = ds_path+'/' + dir
        # assert os.path.exists(ds_dir + "/source.csv")
        # assert os.path.exists(ds_dir + "/target.csv")
        assert os.path.exists(ds_dir + "/rows.txt")
        assert os.path.exists(ds_dir + "/ground truth.csv")

        src_col, target_col = "", ""

        with open(ds_dir + "/rows.txt") as f:
            l = f.readline().strip().split(':')
            src_col = l[0]
            target_col = l[1]
            direction = f.readline().strip()


        pairs = []

        with open(ds_dir + "/ground truth.csv") as f:
            titles = f.readline().strip().split(',')

            if not "source-" + src_col in titles:
                print(ds_dir)

            assert "source-" + src_col in titles
            assert "target-" + target_col in titles

            src_idx = titles.index("source-" + src_col)
            target_idx = titles.index("target-" + target_col)

            if direction.lower() == "target":
                src_idx, target_idx = target_idx, src_idx

            for line in f.readlines():
                items = line.strip().split(',')
                pairs.append((items[src_idx], items[target_idx]))

        if dir_name not in res:
            res[dir_name] = {}
        res[dir_name][typ] = pairs


    return res


def find_joinable_value(inp, examples, targets):

    outs = {}
    for example in examples:
        val = list(example) + [inp]
        model_inp = SRC_PREPARE(val)

        out = transform(model_inp)

        if out in outs:
            outs[out] += 1
        else:
            outs[out] = 1

        if transform2 is not None:
            out = transform2(model_inp)
            if out in outs:
                outs[out] += 1
            else:
                outs[out] = 1

    # srt_dict = {k: v for k, v in sorted(outs.items(), key=lambda item: item[1], reverse=True)}
    srt_lst = [k for k, v in sorted(outs.items(), key=lambda item: item[1], reverse=True)]
    val = srt_lst[0]


    if MATCHING_TYPE == 'edit_dist':
        min_dist = 100000
        min_key = None
        for target in targets:
            dist = nltk.edit_distance(target, val)
            if dist < min_dist:
                min_dist = dist
                min_key = target


        return min_key, val

    elif MATCHING_TYPE == 'exact':
        return val, val if val in targets else None, None

    else:
        raise Exception("Wrong matching type.")


def join(samples, test, number_of_examples):

    assert len(samples) > 1

    if NO_FRAMEWORK:
        examples_for_test = [tuple(random.sample(samples, min(len(samples), number_of_examples)))]
    elif FAST_SAMPLING:
        r = min(len(samples), SAMPLES_PER_EACH_INSTANCE)
        obj_len = len(samples)
        examples_for_test = []
        prems = set()
        tries = 0
        while len(examples_for_test) < min(number_of_examples, int(factorial(obj_len)/factorial(obj_len-r))):
            exp = tuple(random.sample(samples, r))

            if exp not in prems or tries > 10000:
                tries = 0
                examples_for_test.append(exp)
                prems.add(exp)
            else:
                tries += 1
    else:
        # # Use combinations instead of permutations if you want to ignore the order (1,2) <> (2,1)
        # # train_samples = list(itertools.permutations(smpl, r=3))
        prms = itertools.permutations(samples, r=min(len(samples), SAMPLES_PER_EACH_INSTANCE))
        examples_for_test = list(prms)
        random.shuffle(examples_for_test)
        examples_for_test = examples_for_test[:min(len(examples_for_test), number_of_examples)]


    sources = []
    targets = []

    # return [("", "")], [{'inp': "inp", 'gen': "res", 'exp': "out"}]

    joins = []
    predicts = []
    i = 0

    for inp, out in test:
        sources.append(inp)
        targets.append(out)

        res, pred = find_joinable_value(inp, examples_for_test, targets)
        predicts.append({'inp': inp, 'gen': res, 'exp': out, 'pred': pred})
        if res is not None:
            joins.append((inp, res))
        if GPT3_PRINT and MODEL_NAME == "gpt3":
            print(f"    {i} / {len(test)}")
        i += 1


    return joins, predicts


def evaluate(ds_path, number_of_examples_for_join, out_file_path):
    if out_file_path is not None:
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        f = open(out_file_path, 'w')

    # lst = ['FF-dr-name-long-repeat', 'FF-phone-8-short', 'AJ-uk-prime-ministers']
    lst = []
    tables = get_pairs_from_files(ds_path, lst)

    title_str = "id,P,R,F1,correct,len,avg_edit_dist,avg_norm_edit_dist,Time"
    print(title_str)

    if f is not None:
        print(title_str, file=f)

    i = 1
    for table in tables:
        # if table != "AJ-sharif-username-to-email":
        #     print(f"skipping {table}")
        #     continue

        start_time = time.time()
        print(f"{i}/{len(tables)},{table}", end=',')
        i += 1
        smpl = tables[table]['smpl']
        test = tables[table]['test']

        joins, predicts = join(smpl, test, number_of_examples_for_join)

        avg_edit_dist = sum(
            nltk.edit_distance(p['pred'], p['exp']) for p in predicts
        ) / len(predicts)

        avg_norm_edit_dist = sum(
            nltk.edit_distance(p['pred'], p['exp'])/max(len(p['pred']),len(p['exp'])) for p in predicts
        ) / len(predicts)

        correct_cnt = sum(1 if p['pred'] == p['exp'] else 0 for p in predicts)

        je = JoinEval(joins, test)

        taken_time = time.time()-start_time
        # print(str(je.short_str())+","+str(avg_edit_dist)+","+str(taken_time))
        val_str = f"{je.short_str()},{correct_cnt},{len(predicts)},{avg_edit_dist},{avg_norm_edit_dist},{taken_time}"
        print(val_str)

        if f is not None:
            # print(f"{table}," + je.short_str()+","+str(avg_edit_dist)+","+str(taken_time), file=f)
            print(f"{table},{val_str}", file=f)


    if OUT_FILE_PATH is not None:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number-of-examples-for-join', '-n', action='store', type=int, required=False,
                        default=None, help='Number of examples for join')

    parser.add_argument('--use-gpu', '-g', action='store', type=str, required=False,
                        default=None, help='enter Y or y to use cuda if available.')

    parser.add_argument('--no-framework', '-f', action='store', type=str, required=False,
                        default=None, help='enter Y or y to prevent using framework.')

    parser.add_argument('--matching-type', '-t', action='store', type=str, required=False,
                        default=None, help='Matching type for join: [edit_dist, exact] ')

    parser.add_argument('--rel-out-file-path', '-o', action='store', type=str, required=False,
                        default=None, help='Relative output file path, .csv file. leave empty to prevent creating file')

    parser.add_argument('--rel-auto-out-file-dir', '-a', action='store', type=str, required=False,
                        default=None, help='Relative output directory. The file name is auto-generated')

    parser.add_argument('--rel-dataset-path', '-d', action='store', type=str, required=False,
                        default=None, help='Relative dataset folder path')

    parser.add_argument('--rel-model-path', '-p', action='store', type=str, required=False,
                        default=None, help='Relative Model Path')

    parser.add_argument('--model-name', '-m', action='store', type=str, required=False,
                        default=None, help='Model name (from hugging face)')

    parser.add_argument('--model-save-name', '-s', action='store', type=str, required=False,
                        default='', help='Model name to be included in output file')

    args = parser.parse_args().__dict__

    NUMBER_OF_EXAMPLES_FOR_JOIN = args['number_of_examples_for_join'] if args['number_of_examples_for_join'] is not None else NUMBER_OF_EXAMPLES_FOR_JOIN
    MATCHING_TYPE = args['matching_type'] if args['matching_type'] is not None else MATCHING_TYPE
    DS_PATH = BASE_PATH + args['rel_dataset_path'] if args['rel_dataset_path'] is not None else DS_PATH
    DS_NAME = pathlib.PurePath(DS_PATH).name

    if args.get('use_gpu') is not None:
        USE_GPU = True if args['use_gpu'] in ('y', 'Y') else False

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    if args.get('no_framework') is not None:
        NO_FRAMEWORK = True if args['no_framework'] in ('y', 'Y') else False

    if args.get('rel_model_path') is not None:
        MODEL_PATH = BASE_PATH + "/" + args['rel_model_path']

    MODEL_NAME = args['model_name'] if args['model_name'] is not None else MODEL_NAME

    # mdl = os.path.splitext(pathlib.PurePath(MODEL_PATH).name)[0]
    mdl = args['model_save_name']


    if args.get('rel_out_file_path', None) is None:
        if args.get('rel_auto_out_file_dir', None) is None:
            pass  # set defualt value
        else:
            OUT_FILE_PATH = BASE_PATH + f"/{args['rel_auto_out_file_dir']}/joinmdl_{mdl}_DS_{DS_NAME}_{NUMBER_OF_EXAMPLES_FOR_JOIN:02}samp_{MATCHING_TYPE}.csv"
    elif args.get('rel_out_file_path', None) == '':
        OUT_FILE_PATH = None
    else:
        OUT_FILE_PATH = BASE_PATH + "/" + args['rel_out_file_path']

    print(f"dataset path: {DS_PATH}")
    print(f"model path: {MODEL_PATH}")
    print(f"model name: {MODEL_NAME}")
    print(f"output file: {OUT_FILE_PATH}")
    print(f"Using Framework: {(not NO_FRAMEWORK)}")


    if MODEL_NAME == "google/byt5-small" or MODEL_NAME == "google/byt5-base" or MODEL_NAME == "google/byt5-large":
        transform2 = None
        print("Loading Model...")
        transform = byt5_transform

        byt5trainer.MODEL_NAME = MODEL_NAME

        TOKENIZER = byt5trainer.get_tokenizer()

        trained_model = byt5trainer.TransModel.load_from_checkpoint(MODEL_PATH).to(device)
        trained_model.freeze()

        MODEL = trained_model.model

        print("Model Loaded.")
    elif MODEL_NAME == "gpt3":
        transform2 = None
        import openai

        openai.api_key_path = "openai.key"

        OPENAI = openai

        transform = gpt3_transformer
    elif "," in MODEL_NAME:
        tmp = MODEL_NAME.split(",")
        if len(tmp) != 2:
            raise Exception("Model name Wrong")
        if tmp[0] not in ["google/byt5-small", "google/byt5-base", "google/byt5-large"] or tmp[1] != "gpt3":
            raise Exception("Model name Wrong")

        print("Loading Model...")
        transform = byt5_transform
        byt5trainer.MODEL_NAME = tmp[0]
        TOKENIZER = byt5trainer.get_tokenizer()
        trained_model = byt5trainer.TransModel.load_from_checkpoint(MODEL_PATH).to(device)
        trained_model.freeze()
        MODEL = trained_model.model
        print("Model Loaded.")

        import openai
        openai.api_key_path = "openai.key"
        OPENAI = openai
        transform2 = gpt3_transformer
        GPT3_PRINT = False


    else:
        raise Exception("Model name not defined")

    evaluate(DS_PATH, NUMBER_OF_EXAMPLES_FOR_JOIN, OUT_FILE_PATH)


import json
import os
import openai
from openai.error import RateLimitError
import multiprocessing
from tqdm import tqdm

import sys
sys.path.append('..')
from ex_ask import execute_dataset
from llm_utils.askGPT import ASK_GPT
from llm_utils.utils import *
import params

dataset_type = params.__dataset_type

wkeml_score_path = params.__wkeml_score_path
wkpd_score_path = params.__wkpd_score_path


def cutDataset(dataset, save_dir='./main/cut_temp/', num_cut=params.__num_cut):
    keys = list(dataset.keys())
    total_len = len(keys)
    truncation = np.linspace(0, total_len, num_cut+1)
    truncation = [int(t) for t in truncation]
    sub_dicts = []

    for i in range(num_cut):
        sub_dict = {k: dataset[k] for k in keys[truncation[i]:truncation[i+1]]}
        sub_dicts.append(sub_dict)
    
    for idx, sub_dict in enumerate(sub_dicts):
        dumpdata(sub_dict, os.path.join(save_dir, '{}.json'.format(idx+1)))
    return

def combine_res(num_cut=params.__num_cut):
    sup_dicts = []
    for i in range(num_cut):
        sup_dicts.append(loaddata('./main/cut_temp/{}_res.json'.format(i+1)))
    
    combine_dict = {}
    for sup_dict in sup_dicts:
        combine_dict.update(sup_dict)
    
    return combine_dict

class Multiprocess_exdataset(object):
    def __init__(self, keys:str, exdataset_list):

        keyList = keys.split('\n')
        
        self.num_possess = len(exdataset_list)
        every_key_list_len = int(len(keyList)/self.num_possess)
        self.keyListofList = [keyList[i:i + every_key_list_len] for i in range(0, len(keyList), every_key_list_len)]
        print('every key list len: ')
        for i in self.keyListofList:print(len(i))

        self.exdataset_list = exdataset_list

        if dataset_type == 'wikimel':
            self.score_dataset = loaddata(wkeml_score_path)
        elif dataset_type == 'wikidiverse':
            self.score_dataset = loaddata(wkpd_score_path)
        else: raise
    
    def process_func(self, ex_id:int):
        save_path = './main/cut_temp/{}_res.json'.format(ex_id+1)
        temp_path = './main/cut_temp/{}_temp.json'.format(ex_id+1)
        ask_gpt = ASK_GPT(self.keyListofList[ex_id])

        is_record = True if ex_id == 1 else False
        execute_dataset(self.exdataset_list[ex_id], save_path, temp_path, ask_gpt, self.score_dataset, is_record)

        print('{} done'.format(ex_id+1))
        return

    def err_call_back(self, err):
        print(f'error：{str(err)}')
    
    def flow(self):
        pool = multiprocessing.Pool(self.num_possess)

        for ex_id in range(self.num_possess):
            pool.apply_async(self.process_func, args=(ex_id, ), error_callback=self.err_call_back)

        pool.close()
        pool.join()


if __name__ == '__main__':
    print(params.__dataset_type)
    num_cut = params.__num_cut
    keys = params.__keys

    if params.__dataset_type == 'wikimel':
        dataset = loaddata('./dataset_WIKIMEL/WikiMEL_testset.json')
    elif params.__dataset_type == 'wikidiverse':
        dataset = loaddata('./dataset_wikidiverse/WikiDiverse_testset.json')
    else:
        raise

    if len(dataset) > 2000:
        print(f'cut candidate entity to {params.__num_cands}')
        for sample_id, sample in dataset.items(): sample['Candentity'] = sample['Candentity'][:params.__num_cands]
    
    print(len(dataset))

    cutDataset(dataset)

    for sample_id, sample in dataset.items():
        print(len(sample['Candentity']))
        break



    dataset_list = [loaddata('./main/cut_temp/{}.json'.format(i+1)) for i in range(num_cut)]
    print('every slice len：', len(dataset_list[0]))

    operate_test = Multiprocess_exdataset(keys=keys, exdataset_list=dataset_list)

    print('---start---')
    operate_test.flow()
    print('----------DONE----------')


    dumpdata(combine_res(num_cut=num_cut), params.__save_path)
    print(f'save to {params.__save_path}')

    res_set = loaddata(params.__save_path)
    label_set = loaddata('./dataset_WIKIMEL/WikiMEL_testset_label.json' if params.__dataset_type == 'wikimel' else './dataset_wikidiverse/WikiDiverse_testset_label.json')
    r = 0; t = 0
    for sample_id, sample in res_set.items():
        ans = sample['GPTans']
        label = label_set[sample_id]
        if ans == label:
            r += 1
        t += 1
    print('top1-acc:', r, t, r/t)




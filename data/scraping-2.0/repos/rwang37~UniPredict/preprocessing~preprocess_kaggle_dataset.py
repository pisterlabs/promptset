import kaggle
import os
import json
import pandas as pd
import numpy as np
import openai
import time
import torch

from .openai_api import prompt_openai
from .xgb import *
from .utils import *
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from math import floor, log10

curr_path = os.path.join(
    os.path.dirname(os.path.realpath(__name__)),
    'files',
    'data'
)

DEFAULT_DATASET_INDEXING_PATH = 'files/unified/dataset_list/'
DEFAULT_DATASET_SAVING_PATH = 'files/separated/'

class DatasetProcessor():
    def __init__(self, max_dataset_num=2000, metadata_path=None, saving_path=None, debug=False, dataset_info_list_path=None):
        self.debug = debug
        self.save_path = metadata_path if metadata_path else DEFAULT_DATASET_INDEXING_PATH
        self.dataset_path = saving_path if saving_path else DEFAULT_DATASET_SAVING_PATH
        self.max_dataset_num = max_dataset_num
        if dataset_info_list_path:
            self.dataset_info_list = read_json(dataset_info_list_path)
        else:
            self.dataset_info_list = None
           

    def load_dataset_data_from_kaggle(self):
        save_path = self.save_path + 'datasets_after_round_1.json'
        dataset_list = []
        # process at most 2000 pages of responses. Currently, it is more than enough to go through all filtered datasets.
        for i in range(1, 2000):
            if i % 10 == 0 and self.debug:
                print(len(dataset_list))
            temp_list = kaggle.api.dataset_list(
                file_type='csv', 
                tag_ids=13302, 
                max_size=1048576, 
                page=i
            )
            dataset_list.extend(temp_list)
            if len(temp_list) == 0 or len(dataset_list) >= self.max_dataset_num:
                break
        if self.debug:
            print(dataset_list)

        metadata_list = [d.__dict__['ref'] for d in dataset_list]
        save_json(save_path, metadata_list)
        self.dataset_info_list = metadata_list
        return metadata_list
    
    def save_dataset_data(self):
        for item in self.dataset_info_list:
            save_path = item.replace('/', '-')
            download_path = self.dataset_path + save_path

            if self.debug:
                print(f'downloading and saving file to {download_path}')
            try:
                kaggle.api.dataset_metadata(
                    item, 
                    path=download_path
                )

                kaggle.api.dataset_download_files(
                    item, 
                    path=download_path, 
                    unzip=True
                )
            except Exception as e:
                if self.debug:
                    print('failed')
                    print(e)
                pass
        
    def preprocess_all_metadata(self, pivot=0):
        count = 0
        dataset_info_list = self.dataset_info_list[pivot:]
        for item in dataset_info_list:
            if self.debug:
                print(f'Making metadata for {item}. Current progress: {count} metadata saved')
            item_path = self.dataset_path + item.replace('/', '-')
            try:
                preprocess_metadata(item_path)
                count += 1
            except openai.error.RateLimitError:
                # retry until no ratelimit error
                if self.debug:
                    print('retrying')
                result = None
                while result is None:
                    time.sleep(10)
                    try:
                        preprocess_metadata(item_path)
                        result = 'worked'
                    except openai.error.RateLimitError:
                        pass
                    except:
                        break
            except Exception as e:
                if self.debug:
                    print(f'failed: {e}')
                pass
            if self.debug:
                print('\n\n')
        if self.debug:
            print(count)

    def preprocess_all_data(self):
        save_path = self.save_path
        processed_dataset_list = []
        for item in self.dataset_info_list:
            dataobj = DataObject(item)
            if dataobj.get_availability():
                processed_dataset_list.append(item)
        save_json(save_path + 'datasets_after_round_2.json', processed_dataset_list)

class DataObject():
    def __init__(self, name, path=None, from_preprocessed=False, output_type='Default'):
        self.availability = True
        if not path:
            self.location = DEFAULT_DATASET_SAVING_PATH + name.replace('/', '-')
        else:
            self.location = path + name.replace('/', '-')

        if not from_preprocessed:
            try:
                self.preprocess_data()
            except Exception as e:
                print(e)
                self.availability = False
        else:
            self.load_from_preprocessed()

        self.reshape_output(output_type=output_type)
    
    def load_from_preprocessed(self):
        train_path = os.path.join(self.location, 'train_set.pt') 
        test_path = os.path.join(self.location, 'test_set.pt') 
        self.train = torch.load(train_path)
        self.test = torch.load(test_path)
    
    def preprocess_data(self):
        file_path = self.location

        # filter out the unqualified datasets
        files_lst = []
        for root, dirs, files in os.walk(file_path):
            files_lst.extend(files)
        assert 'metadata.json' in files_lst, f'Preprocessed metadata not included in this folder: {files_lst}, {file_path}'

        # define paths to datasets and metadata files
        files_lst.remove('dataset-metadata.json')
        files_lst.remove('metadata.json')
        csv_path = [item for item in files_lst if '.csv' in item][0]
        dataset_path = os.path.join(file_path, csv_path)
        metadata_path = os.path.join(file_path, 'metadata.json')

        # open files and extract values
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        dataset = pd.read_csv(dataset_path)
        target_name = metadata['target']
        bin_lst = metadata['bins']
        label_lst = metadata['labels']
        annotations = metadata['metadata']
        print(target_name)

        # prepare samples (x), column values, labels (y) and annotations (dataset description/metadata)
        samples = dataset.drop(target_name, axis=1).round(4)
        col = samples.columns.to_list()
        if metadata['bins'] != 'N/A' and dataset[target_name].nunique() > 10:
            segs = [0] + bin_lst + [np.inf]
            labels = pd.cut(dataset[target_name], bins=segs, labels=label_lst)
        else:
            labels = dataset[target_name]
        annotations = [annotations] * len(samples)

        # preprocessing
        xgb_baseline, prompts, output = transform_data(samples, col, labels, annotations)
        
        # rd: raw_data
        # ro: raw_output
        # p: prompts
        # a: annotations
        # l: label_cats
        # o: outputs
        rd_train, rd_test, ro_train, ro_test, p_train, p_test, a_train, a_test, l_train, l_test, o_train, o_test = train_test_split(
            xgb_baseline[0],
            xgb_baseline[1],
            prompts[0],
            prompts[1],
            prompts[2],
            output,
            test_size=0.1,
            random_state=42
        )
        # overwrite output_train so that it only uses the data from the train set
        # prevent leak
        augmentor = DataAugmentor(rd_train, ro_train)
        o_train, auc = augmentor.generate_label_prompt()
        train = ((rd_train, ro_train), (p_train, a_train, l_train), o_train)
        test = ((rd_test, ro_test), (p_test, a_test, l_test), o_test)

        self.train = train
        self.test = test

        self.save_train()
        self.save_test()
    
    def save_train(self, filename='train_set.pt'):
        path = os.path.join(self.location, filename)
        torch.save(self.train, path)
    
    def save_test(self, filename='test_set.pt'):
        path = os.path.join(self.location, filename)
        torch.save(self.test, path)

    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test
    
    def get_availability(self):
        return self.availability

    def reshape_output(self, output_type='Default'):
        if output_type == 'TabLLM':
            output = [f'class {int(item)}' for item in self.train[0][1]]
            self.train = (self.train[0], self.train[1], output)
        elif output_type == 'Ablation':
            temp = self.train[0][1]
            temp = temp.squeeze(-1)
            onehot = np.zeros((temp.size, temp.max() + 1))
            onehot[np.arange(temp.size), temp] = 1
            output = serialize_output(onehot)
            self.train = (self.train[0], self.train[1], output)
        elif output_type == 'Default':
            pass
        else:
            raise NotImplementedError



def preprocess_metadata(path):
    file_path = path

    files_lst = []
    print(file_path)
    for root, dirs, files in os.walk(file_path):
        files_lst.extend(files)
    assert 'dataset-metadata.json' in files_lst, f'Metadata does not exist: {files_lst}'

    metadata_path = os.path.join(file_path, 'dataset-metadata.json')
    files_lst.remove('dataset-metadata.json')
    dataset_path = os.path.join(file_path, [item for item in files_lst if 'csv' in item][0])

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    metadata = metadata['description']

    meta, bin_lst, label_lst, target = extract_metadata_from_data(dataset_path, metadata)

    save_path = os.path.join(file_path, 'metadata.json')
    with open(save_path, 'w+') as f:
        json.dump({
            "target": target,
            "metadata": meta,
            "bins": bin_lst,
            "labels": label_lst
        }, f, indent=4)


def extract_metadata_from_data(path, metadata):
    # 1. read file
    data = pd.read_csv(path)
    col = str(data.columns.to_list())

    prompt = (
        "The following is the metadata of a tabular dataset. Return the information for:\n" 
        "    1. the target of the dataset. If no target exists, choose one from the column as target for the dataset to classify.\n"
        "    2. the features and their explanations, or N/A if there are no explanations. Replace all hyphens and/or underscores with spaces.\n\n"
        "Give your output in json. The following is an example output:\n"
        '{\n'
        '    "target": "Age",\\n'
        '    "metadata": "The target of the dataset is Age. \\n Features and their explanations:\\n    gender: an animal\'s gender.\\n    weight: an animal\'s actual weight, in kg." \\n '
        '}\n\n'
        "Do NOT respond anything else than the needed information. Make it brief but informative." 
        "Your responses should only be code, without explanation or formatting.\n\n"
        f"columns:{col}\n\n" 
        f"metadata:{metadata}\n"
        "Provide your response in stringfied JSON format." 
    )

    response = prompt_openai(prompt)
    print(repr(response))
    response = json.loads(response)
    target = response['target']
    metadata = response['metadata']
    if is_numeric_dtype(data[target]) and (data[target] != 0).all():
        dataset_overview = data[target].describe().apply(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        seg0 = dataset_overview['25%']
        seg1 = dataset_overview['50%']
        seg2 = dataset_overview['75%']

        seg_str0 = f'<{seg0}'
        seg_str1 = f'{seg0} - {seg1}'
        seg_str2 = f'{seg1} - {seg2}'
        seg_str3 = f'>{seg2}'
        return metadata, [seg0, seg1, seg2], [seg_str0, seg_str1, seg_str2, seg_str3], target
    else:
        return metadata, "N/A", "N/A", target


def transform_data(data, col, labels, annotations):
    prompt = data_to_prompt(data, col)

    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    # print(cat_dict)
    print(len(prompt))
    assert len(prompt) < 20000, 'Too many samples in the file. Skipping...'

    augmentor = DataAugmentor(data, labels, col)
    outputs, auc = augmentor.generate_label_prompt()
    label_cat = label_to_prompt(cat_dict, len(prompt))
    
    # prompts, outputs, annotations, label_cats
    raw_data, raw_label = numericalize(data, labels, col)

    return (raw_data, raw_label), (prompt, annotations, label_cat), (outputs)

import pandas as pd
from collections import Counter
import pymongo
import json
import os
import tiktoken
import openai
import backoff
from dotenv import load_dotenv
load_dotenv()

openai.organization = os.getenv("ORG_ID")
openai.api_key = os.getenv("API_KEY")

DB = pymongo.MongoClient(host='127.0.0.1', port=27017)['freefuzz-tf']


def read_txt(fname):
    with open(fname, "r") as fileReader:
        data = fileReader.read().splitlines()
    return data


def write_list_to_txt4(data, filename):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(data+'\n')


def calculate_rule_importance(data):
    element_frequency = Counter(data['Anomaly'].values.flatten())
    total_elements = len(data['Anomaly'].values.flatten())
    element_importance = {element: frequency /
                          total_elements for element, frequency in element_frequency.items()}
    return sorted(element_importance.items(), key=lambda x: x[1], reverse=True)


def main():
    data = pd.read_csv('data/TF_RECORDS.csv', sep=',', encoding='utf-8')
    weights_ = calculate_rule_importance(data)
    unique_types = data['Category'].unique()
    unique_types = list(unique_types)
    for dtype in unique_types:
        anomalies = data.loc[data['Category'] == dtype, 'Anomaly']
        anomalies_unique = list(anomalies.unique())
        print('')


def gpt_conversation(prompt, model="gpt-3.5-turbo"):

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response



history_based_partitions = {
    "<integer>": 
    {
        'Integer argument that have negative values': 'max_output_size=-1,',
        'Integer argument that have zero values': 'batch_size = 0',
        'Integer argument that have large values': 'out_dim = 1250999896764',
        'Integer argument that have negative and large values': 'out_dim = -1250999896764',
        'Integer argument that are empty': 'dim= []',

    },
    "<float>": {
        'Float argument that have negative values': 'max_output_size=-1.9',
        'Float argument that have zero values': 'max_output_size=-0.0',
        'Float argument that have larges': 'max_output_size=1250999896764.1',
        'Float argument that have negative and large values': 'max_output_size=-1250999896764.1',
        'Empty float arguments': 'dim= []',
    },
    '<string: 2>':
    {
        "Invalid string": '流星暴雨',
        "Empty string": 'logdir=''',
    },
    '<Python list: 6>': {
        'A list containing large integer or float elements': 'limits=[10.0,1e20]',
        'A list containing integer or float zero elements': 'arg = [3, 1, 0, 3]',
        'A list containing negative integer or float elements': 'row_pooling_sequence=[-10,1,2,3]',
        'An empty list': 'indices=[[],[],[]],',
        'A list that contains invalid string values': 'handle=["\x12\x1a\x07"]',
        'A list containing nan elements': 'limits=[np.nan,1e20]',
        'A list containing None elements': 'limits=[None,1e20]',
        'A list containing invalid elements':'handle=["\x12\x1a\x07", 1, 2]'
    },
    '<Python tuple: 5>': {
        'A tuple containing large integer or float elements': 'limits=(10.0,1e20)',
        'A tuple containing integer or float zero elements': 'arg = (3, 1, 0, 3)',
        'A tuple containing negative integer or float elements': 'row_pooling_sequence=(-10,1,2,3)',
        'An empty tuple': 'indices=((),(),())',
        'A tuple that contains invalid string values': 'handle=("\x12\x1a\x07", 1, 2)',
        'A tuple with nan elements': 'limits=(np.nan,1e20)',
        'A tuple with None elements': 'limits=(None,1e20)',
        'A tuple with invalid elements':'limits=("\x12\x1a\x07",1e20)'
    },
    '<tensor: 9>': {
        "Input tensor that is not scalars": 'num_bits = tf.constant([], shape=[0], dtype=tf.int32)',
        "tensors with large values and shape": 'arg_0=tf.random.uniform(shape=(4,), dtype=tf.int32, maxval=65536)',
        'tensors with negative shapes and values': 'orig_input_shape = tf.constant(-536870912, shape=[4], dtype=tf.int32)',
        'tensors with scalar values': 'dataset = tf.data.Dataset.range(3)',
        'tensors with nan values': 'x = torch.tensor(float(np.nan)).cuda()',
        'tensors with zero values and ranks': 'b = torch.IntTensor([0,1])',
        'tensors with empty values': 'total_length = torch.full([], -9937, dtype=torch.int64, requires_grad=False)',
    },
}

def create_prompt_generate_samples_for_history_partitions(partition, arg, ex):
    prompt_ = f"""
    You are an experienced software developer. 
    You are great at understanding software securities and bugs that are caused by feeding malicious inputs to Python APIs. 
    When you don't know how to generate malicious samples for fuzzing, you admit that you don't know. 

    Your task is to generate malicious samples in a systematic manner for each partition {partition} of the argument type {arg} for API-level fuzzing.
    
    Here is an example {ex}.
    Generate as many malicious samples as you can. Do not generate duplicate samples.
    
    Please note that API level fuzzing is for TensorFLow and PyTorch.
    
    Please output the samples in given json format:
        
    <answer json start>,
    "Sample 1":"Sample 1",
    "Sample 2":"Sample 2",
    ...
    "Sample n":"Sample n",

    """
    return prompt_

def create_prompt_convert_partition_to_code(partition, arg, api):
    prompt_ = f"""
    You are an experienced TensorFlow backend software developer. You are also great at converting natural language to source code. When you don't know how convert a natual languae to code, you admit that you don't know.
    
    Your task is to convert the given parition{partition} of the argument {arg} of the API {api} to Python code example.
    
    Please generate only one example code for the parition. 

    Please output the partitions in given json format:
        
    <answer json start>,
    "Partition":"Example python client code",
    """
    return prompt_

def create_prompt_fix_sugesstion():
    prompt_ = f"""
    You are an experienced software developer in API-level fuzz testing. You are great at understanding software security and bugs that are caused by feeding malicious inputs to APIs. When you don't know how to do input space partitioning, you admit that you don't know. 

    Your task is to perform input space partitioning for each parameter in {api_name}. The arguments and their values are:
    Arguments: {record}
    
    Generate as many partitions as you can for each argument.  
    
    Please output the partitions in given json format:
        
    <answer json start>,
    "Partition 1":"Explain partition 1",
    "Partition 2":"Explain partition 2",
    ...
    "Partition n":"Explain partition n",

    """

    return prompt_

def create_prompt_argument_partitioning(api_name, record):
    prompt_ = f"""
    You are an experienced software developer in API-level fuzz testing. You are great at understanding software security and bugs that are caused by feeding malicious inputs to APIs. When you don't know how to do input space partitioning, you admit that you don't know. 

    Your task is to perform input space partitioning on the argument {record} of the API {api_name}.
    
    Generate as many partitions as you can for each argument.  
    
    Please output the partitions in given json format:
        
    <answer json start>,
    "Partition 1":"Explain partition 1",
    "Partition 2":"Explain partition 2",
    ...
    "Partition n":"Explain partition n",

    """
    return prompt_


def create_prompt_type_partitioning(arg_type):

    prompt_type = f"""
    You are an experienced software developer in fuzz testing as well as API-level testing. You are great at understanding software security and bugs that are caused by malicious inputs to APIs. When you don't know how to do input space partitioning, you admit that you don't know. 

    Your task is to perform input space partitioning for a {arg_type} argument. Generate as many partitions as you can.

    Please output the partitions in given json format:
    
    <answer json start>
    "Partition 1":"Python code for partition 1",
    "Partition 2":"Python code for partition 2",
    ...
    "Partition n":"Python code for partition n",

    """
    return prompt_type


def get_api_seed(api_name):
    record = DB[api_name].aggregate([{"$sample": {"size": 1}}])
    if not record.alive:
        print(f"NO SUCH API: {api_name}")
        assert (0)
    record = record.next()
    record.pop("_id")
    assert ("_id" not in record.keys())
    return record


def exec_input_sp_type():
    lib_name = 'tf'
    rules_path = f"parition_rules/{lib_name}_type_patitions.json"
    type_lists = ['ArgType.Tensor', 'ArgType.INT', 'ArgType.STR',
                  'ArgType.FLOAT', 'ArgType.LIST', 'ArgType.TUPLE']
    for arg_type in type_lists:
        prompt_ = create_prompt_type_partitioning(arg_type)
        t_count = get_token_count(prompt_)
        if t_count <= 4097:
            conversations = completions_with_backoff(prompt_)
            rule_ = conversations.choices[0].message.content

            try:
                x = json.loads(rule_)
                x.update({'Arg type': arg_type})

                with open(rules_path, "a") as json_file:
                    json.dump(x, json_file, indent=4)
                    json_file.write(',')
                    json_file.write('\n')
            except Exception as e:
                print(e)
        else:
            print("Your messages exceeded the limit.")

        print('')


def get_token_count(string):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    num_tokens = len(encoding.encode(string))

    return num_tokens


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(prompt, model='gpt-3.5-turbo'):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response


def load_json(_path):
    f = open(_path)
    data = json.load(f)
    return data

def argument_partition_l2():
    tf_parts = load_json("output/tf_partitions_per_api.json")
    for item in tf_parts:
        api_holder = {}
        for api_k, api_args in item.items():
            param_holder = {}
            for param_k, param_v in api_args.items():
                part_holder = {}
                for part_k, part_v in param_v.items():
                    prompt_ = create_prompt_convert_partition_to_code(part_v, param_k, api_k)
                    conversations = completions_with_backoff(prompt_)
                    rule_ = conversations.choices[0].message.content
                    part_holder[part_k] = rule_
                param_holder[param_k] = part_holder
            api_holder[api_k] = param_holder
                    

def argument_partition_l1():
    lib_name = 'tf'
    rules_path = f"output/{lib_name}_partitions_per_api.json"

    hisotry_file = f"logs/parsed_apis.txt"

    if not os.path.exists(hisotry_file):
        f1 = open(hisotry_file, 'a')

    hist = read_txt(f'logs/parsed_apis.txt')
    for api_name in DB.list_collection_names():
        if api_name not in hist:
            write_list_to_txt4(api_name, hisotry_file)
            print(api_name)
            record = get_api_seed(api_name)
            #record.pop("source")
            arg_per_part = {}
            for k, v in record.items():
                if k != 'input_signature' and k != 'output_signature':
                    record_json = json.dumps(v)
                    prompt_ = create_prompt_argument_partitioning(api_name, record_json)
                    t_count = get_token_count(prompt_)
                    if t_count <= 4097:
                        conversations = completions_with_backoff(prompt_)
                        rule_ = conversations.choices[0].message.content
                        try:
                            x = json.loads(rule_)
                            arg_per_part[k] = x
                        except Exception as e:
                                print(e)
            x = {}
            x[api_name] = arg_per_part
            with open(rules_path, "a") as json_file:
                json.dump(x, json_file, indent=4)
                json_file.write(',')
                json_file.write('\n')
   

def gen_samples_for_history_partitions():
    rules_path = "/media/nimashiri/DATA/vsprojects/llmrules/output/general_part_samples.json"
    
    for k, v in history_based_partitions.items():
        type_holder = {}
        part_holder = {}
        for part_k, part_v in v.items():
            print(f"Working on {k}:{part_k}")
            prompt_ = create_prompt_generate_samples_for_history_partitions(part_k, k, part_v)
            conversations = completions_with_backoff(prompt_)
            rule_ = conversations.choices[0].message.content
            rule_ = rule_.replace('{', '')
            rule_ = rule_.replace('}', '')
            rule_ = rule_.split('\n')
            rule_ = [x for x in rule_ if x != '']
            part_holder[part_k] = rule_
        type_holder[k] = part_holder
        
        with open(rules_path, "a") as json_file:
            json.dump(type_holder, json_file, indent=4)
            json_file.write(',')
            json_file.write('\n')
   

if __name__ == '__main__':
    gen_samples_for_history_partitions()

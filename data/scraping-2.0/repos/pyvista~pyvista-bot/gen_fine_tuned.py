"""
Example: Generate a pyvista source function using single shot training.

"""
from multiprocessing.pool import Pool

from misc import remove_comments_and_docstrings
import csv
import warnings
from tqdm import tqdm
import inspect
import openai
from get_funcs import get_funcs, get_body
from vtk_info import get_cls_info
from transformers import GPT2TokenizerFast
from pyvista import _vtk
import pickle
import os

from key import API_KEY

openai.api_key = API_KEY

QUERY_VTK = True
LOAD_OLD = False

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

vtk_classes = []
for attr_str in dir(_vtk):
    item = getattr(_vtk, attr_str)
    # print(item)
    if item.__class__ == type and attr_str.startswith('vtk'):
        vtk_classes.append(item)

completions_filename = 'completions.p'
if os.path.isfile(completions_filename) and LOAD_OLD:
    with open(completions_filename, 'rb') as fid:
        completions = pickle.load(fid)
    print('loaded', len(completions), 'completions')
else:
    completions = {}

def gen_completion(vtk_cls):
    """Generate a completion."""
    vtk_class_name = vtk_cls.__name__

    if vtk_class_name in completions:
        print('Already completed {vtk_class_name}')
        return

    try:
        pv_funcs, pv_func = get_funcs(vtk_class_name, find_test=False)
    except Exception as e:
        print(f'Failed to get function:\n{e}')
        return

    pv_source = remove_comments_and_docstrings(inspect.getsource(pv_func))

    if QUERY_VTK:
        try:
            vtk_cls_info = get_cls_info(vtk_class_name)
        except Exception as e:
            print(f'{vtk_class_name}\nTreating warning as exception "{str(e)}"')
            return
    else:
        vtk_cls_info = {'cls_name': '',
                        'short_desc': '',
                        'long_desc': '',
                        'fnames': '',
                        }

    prompt = f"""{vtk_cls_info['cls_name']}
{vtk_cls_info['short_desc']}
{vtk_cls_info['long_desc']}
{vtk_cls_info['fnames']}
"""

    completion = f""" # {pv_func.__module__}
# {pv_func.__qualname__}

{pv_source}##
"""

    tok_prompt = tokenizer(prompt)
    tok_completion = tokenizer(completion)

    print(
        f'{vtk_class_name:35s}',
        f'{str(pv_func.__class__):30s}',
        len(tok_prompt['input_ids']),
        len(tok_completion['input_ids']),
    )
    return vtk_class_name, {'prompt': prompt, 'completion': completion}

submit = [vtk_class for vtk_class in vtk_classes if vtk_class.__name__ not in completions]

with Pool(8) as pool:
    out = pool.map(gen_completion, submit)

for item in out:
    if item is None:
        continue
    vtk_class_name, item_info = item
    completions[vtk_class_name] = item_info

# save completions to disk
with open(completions_filename, 'wb') as fid:
    pickle.dump(completions, fid)


# writing to csv file
filename = 'completions.csv'
with open(filename, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['prompt', 'completion'])

    for obj in completions:
        prompt = completions[obj]['prompt']
        completion = completions[obj]['completion']
        tok = tokenizer(prompt + completion)

        if len(tok) < 4097:
            writer.writerow([prompt, completion])


def print_completion(vtk_cls):
    print(vtk_cls)
    print(completions[vtk_cls]['prompt'])
    print(completions[vtk_cls]['completion'])


# print_completion('vtkLight')

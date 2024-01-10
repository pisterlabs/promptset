import ujson as json
import jsonlines
from key import *
from tqdm import tqdm, trange
from multiprocessing import Pool, freeze_support, RLock, Manager, Process
import os
import pickle
import chromadb
from pathlib import Path
import re
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import utils as U
import random
import networkx as nx

from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('/data1/xinhuajian/ColossalAIChat/output/WizardCoder-1B-V1.020231015_144307/epoch0-step14000/model', fast_tokenizer=True, cache_dir='/data1/xinhuajian/cache')

tokenizer.sep_token = tokenizer.mask_token = tokenizer.cls_token = tokenizer.eos_token

def remove_line_numbers(seq):
    result = []
    for line in seq.split('\n'):
        if m := lean_code_regex.match(line):
            result.append(line[m.span()[1] : ])
        else:
            result.append(line)
    return '\n'.join(result)

num_processes = 30

def add_item(result_queue, total):
    # chroma_client = chromadb.PersistentClient(path=f"multi_level_library/skill/vectordb")
    # vectordb = chroma_client.create_collection(name="skill_library")
    dataset = []
    with trange(total, position=0) as tbar:
        while True:
            item = result_queue.get()
            if item is None:
                with open('small_temp.pickle', 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                break
            dataset.append(item)
            # metadatas, embeddings, texts, ids = item
            # vectordb.add(metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids)
            tbar.update()

lean_code_regex = re.compile(r'#\d+ ')

def remove_line_numbers(seq):
    result = []
    for line in seq.split('\n'):
        if m := lean_code_regex.match(line):
            result.append(line[m.span()[1] : ])
        else:
            result.append(line)
    return '\n'.join(result)

def get_env(env):
    buffer = []
    namespaces = []
    opens = []
    for _, _, t, name in env:
        if t == 'namespace':
            namespaces.append(name)
            buffer.insert(0, name.split()[-1])
        elif t == 'open':
            opens.append(name)
    return namespaces, [f'end {item}' for item in buffer], opens

base_prompt = '''Below is an instruction that describes a task.
Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n'''

des_instruction = '''Analyze a Lean 3 declaration (e.g., definition, theorem) and provide a clear, detailed description of its mathematical meaning, including the represented concepts, principles, and rules, in language accessible to mathematics students and professionals.'''

itp_instruction = '''You will receive a theorem or lemma, an incomplete proof of it, and the corresponding proof state at specified positions. Each line of the incomplete proof code is prefixed with #n, indicating the nth line of the code. Please note that #n is supplementary information and not part of the code.

Your task is to translate the given proof state into natural language, explaining the premises and proof goals without directly copying the content of the provided proof state. 
'''

dft_instruction = '''You will receive a theorem or lemma, an incomplete proof of it, and the corresponding proof state at specified positions. Each line of the incomplete proof code is prefixed with #n, indicating the nth line of the code. Please note that #n is supplementary information and not part of the code.

Your task is to develop a strategy for a valid, effective step, and explain it in three aspects: identify used math concepts, assess strategies/skills, and explain its role in the overall proof.  You should respond in the following format:

<format>
- Overall Planning:
...
- Identification of Mathematical Concepts:
...
- Analysis of Mathematical Strategies and Skills:
...
- Contribution to Overall Proof:
...
</format>
'''

imp_instruction = '''In Lean 3, the placeholder 'sorry' is used in incomplete proofs. You'll receive a theorem/lemma, an incomplete proof, and proof state positions to be replaced. Each line of the incomplete proof code is prefixed with #n, indicating the nth line of the code. Please note that #n is supplementary information and not part of the code.

Your task is to complete the proof, using 'sorry' for uncertain details. You should respond in the following format:
<format>
```lean3
...
```
</format>
'''

total_instruction = """You will receive a theorem or lemma along with an incomplete proof of it. Each line of the incomplete proof code is prefixed with #n, indicating the nth line of the code. Please be aware that #n is only supplementary information and is not a part of the actual code. This task consists of three sequential subtasks:

1. **Translation of Proof States to Natural Language**:
   - Given the corresponding proof state at specified positions, your first subtask is to translate this proof state into natural language. Your aim is to explain both the premises and proof goals without directly copying the content of the provided proof state.

2. **Development of Proof Strategy**:
   - Analyze the proof and develop a strategy for a valid, effective step. Your explanation should encompass the following aspects:
     - Overall Planning
     - Identification of Mathematical Concepts
     - Analysis of Mathematical Strategies and Skills
     - Contribution to the Overall Proof

   Use the provided format for your response:
   ```
   - Overall Planning: ...
   - Identification of Mathematical Concepts: ...
   - Analysis of Mathematical Strategies and Skills: ...
   - Contribution to Overall Proof: ...
   ```

3. **Completion in Lean 3**:
   - Based on the analysis, complete the proof in Lean 3. In places where you're uncertain of the details, use the placeholder 'sorry'. 

Remember, each subtask builds upon the previous one, so ensure consistency and coherence in your responses. You should respond in the following format:

<format>
- Translation of Proof States to Natural Language:
...
- Overall Planning:
...
- Identification of Mathematical Concepts:
...
- Analysis of Mathematical Strategies and Skills:
...
- Contribution to Overall Proof:
...
- Completion in Lean 3:
```lean3
...
```
</format>
"""

wo_completion_instruction = """You will receive a theorem or lemma along with an incomplete proof of it. Each line of the incomplete proof code is prefixed with #n, indicating the nth line of the code. Please be aware that #n is only supplementary information and is not a part of the actual code. This task consists of two sequential subtasks:

1. **Translation of Proof States to Natural Language**:
   - Given the corresponding proof state at specified positions, your first subtask is to translate this proof state into natural language. Your aim is to explain both the premises and proof goals without directly copying the content of the provided proof state.

2. **Development of Proof Strategy**:
   - Analyze the proof and develop a strategy for a valid, effective step. Your explanation should encompass the following aspects:
     - Overall Planning
     - Identification of Mathematical Concepts
     - Analysis of Mathematical Strategies and Skills
     - Contribution to the Overall Proof

   Use the provided format for your response:
   ```
   - Overall Planning: ...
   - Identification of Mathematical Concepts: ...
   - Analysis of Mathematical Strategies and Skills: ...
   - Contribution to Overall Proof: ...
   ```

Remember, each subtask builds upon the previous one, so ensure consistency and coherence in your responses. You should respond in the following format:

<format>
- Translation of Proof States to Natural Language:
...
- Overall Planning:
...
- Identification of Mathematical Concepts:
...
- Analysis of Mathematical Strategies and Skills:
...
- Contribution to Overall Proof:
...
</format>
"""

def simplify_lean_code(code):
    import re
    from collections import defaultdict
    lines = code.split('\n')
    simplified_lines = []  # 存储简化后的代码行

    for line in lines:
        ns_match = re.match(r'\s*namespace ([^\s]+)', line)
        end_match = re.match(r'\s*end ([^\s]+)', line)
        if ns_match:
            ns_name = ns_match.group(1)
            # 查找simplified_lines中最后一个非空行
            for i in range(len(simplified_lines) - 1, -1, -1):
                if simplified_lines[i].strip():
                    last_non_empty_line = simplified_lines[i]
                    break
            else:
                last_non_empty_line = ''

            end_match = re.match(r'\s*end ' + re.escape(ns_name), last_non_empty_line)
            if end_match:
                # 如果找到匹配的end语句，删除它
                del simplified_lines[i]
            else:
                # 否则，添加当前的namespace语句
                simplified_lines.append(line)
        elif end_match:
            # 添加当前的end语句
            simplified_lines.append(line)
        else:
            # 对于非namespace和end语句，直接添加到简化后的代码行
            simplified_lines.append(line)

    lines = simplified_lines
    simplified_lines = []  # 存储简化后的代码行

    namespace_stack = []  # 栈用来存储当前的namespace层级
    opened_namespaces = defaultdict(set)  # 用来存储在每个namespace层级中已经打开的命名空间

    for line in lines:
        ns_match = re.match(r'\s*namespace ([^\s]+)', line)
        end_match = re.match(r'\s*end ([^\s]+)', line)
        open_match = re.match(r'\s*open ([^\s]+)', line)

        if ns_match:
            ns_name = ns_match.group(1)
            namespace_stack.append(ns_name)
            simplified_lines.append(line)
        elif end_match:
            if namespace_stack:  # 防止空栈
                namespace_stack.pop()
            simplified_lines.append(line)
        elif open_match:
            open_ns = open_match.group(1)
            should_add = True
            # 检查是否在当前namespace层级或上层中已经打开了这个命名空间
            for ns in reversed(namespace_stack):
                if open_ns in opened_namespaces[ns]:
                    should_add = False
                    break
            if should_add:
                # 如果没有打开过这个命名空间，则添加这个open语句并更新opened_namespaces
                opened_namespaces[namespace_stack[-1]].add(open_ns) if namespace_stack else None
                simplified_lines.append(line)
        else:
            # 对于非namespace, end和open语句，直接添加到简化后的代码行
            simplified_lines.append(line)

    return re.sub(r'\n\n+', '\n\n', '\n'.join(simplified_lines))

def compute_replacement(before, after):
    # 找到第一个不同字符的位置
    start_diff = 0
    while start_diff < min(len(before), len(after)) and before[start_diff] == after[start_diff]:
        start_diff += 1
    
    # 向前移动到完整的符号或词的开始
    while start_diff > 0 and not re.match(r'[\s\W]', before[start_diff:]):
        start_diff -= 1

    # 找到最后一个不同字符的位置
    end_diff_before = len(before)
    end_diff_after = len(after)
    while (end_diff_before > start_diff and end_diff_after > start_diff and
           before[end_diff_before - 1] == after[end_diff_after - 1]):
        end_diff_before -= 1
        end_diff_after -= 1
    
    # 向后移动到完整的符号或词的结束
    while (end_diff_before < len(before) and not re.match(r'[\s\W]', before[end_diff_before - 1:])):
        end_diff_before += 1
    while (end_diff_after < len(after) and not re.match(r'[\s\W]', after[end_diff_after - 1:])):
        end_diff_after += 1

    # 提取替换前和替换后的内容
    replaced_from = before[start_diff:end_diff_before]
    replaced_to = after[start_diff:end_diff_after]

    return (replaced_from, replaced_to)

multi_line_regex = re.compile(r'\n\n+')

def main():
    # selected_path = ['data', 'algebra', 'analysis', 'field_theory', 'geometry', 'linear_algebra', 'init.data', 'logic', 'number_theory', 'topology.instances', 'order.filter', 'order.well_founded']
    total_len = 0
    # decl_description = set()
    proof_state_interpretation = set()
    proof_step_drafting = set()
    proof_step_implementation = set()
    # manager = Manager()
    # result_queue = manager.Queue()
    with open('description_temp.json') as f:
        desp = json.load(f)
    with open('export_db.json', 'r') as f:
        decls = json.load(f)
    file_graph = nx.read_gexf("import.gexf")
    with open("dependency_graph.pickle", "rb") as f:
        decl_graph = pickle.load(f)
    topological_order = list(nx.topological_sort(decl_graph))
    topological_order.reverse()
    with open('open_env.json') as f:
        env = json.load(f)
    init_end = json.loads(open('decl_init_end.json').read())
    with jsonlines.open('multilevel_proofstep.jsonl') as reader, trange(0) as tbar:
        for i in reader.iter():
            # if not any(decls[i[1]]['filename'].startswith(j) for j in selected_path):
            #     continue
            codes = [i for i in re.findall(r'```Lean3(.*?)```', i[6], re.DOTALL)]
            codes[1] = remove_line_numbers(codes[1]).strip('\n')
            if not len(codes) == 2 or re.sub(r'\s', '', codes[0]) in ['bysorry', 'beginsorryend', 'beginsorry,end']:
                continue
            i.append(desp[i[1]])
            precedent_files = []
            try:
                for pf in file_graph.successors('mathlib:'+decls[i[1]]['filename']):
                    if pf.startswith('mathlib:'):
                        precedent_files.append(pf[len('mathlib:') : ])
            except:
                pass
            new_premises = []
            retrieved_premises = []
            _, proof_step = compute_replacement(remove_line_numbers(i[2]).strip('\n'), codes[1])
            if not proof_step:
                continue
            for premise in [i for i in decl_graph.successors(i[1])]:
                if decls[premise]['filename'] in env and premise in env[decls[premise]['filename']]:
                    if 'mathlib' not in decls[premise]['local_filename'] or premise.split('.')[-1] not in proof_step:
                        continue
                    namespaces, ends, opens = get_env(env[decls[premise]['filename']][premise])
                    if random.random() > 0.5:
                        if decls[premise]['kind'] == 'theorem' and premise in init_end:
                            new_premises.append((topological_order.index(premise),'\n\n'.join(namespaces + opens + [init_end[premise][0]] + ends)))
                        else:
                            new_premises.append((topological_order.index(premise),'\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
                    else:
                        if decls[premise]['kind'] == 'theorem' and premise in init_end:
                            if random.random() > 0.8:
                                retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
                            else:
                                retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [init_end[premise][0]] + ends)))
                        else:
                            retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
            new_premises = [i[1] for i in sorted(new_premises)]
            retrieved_premises = [i[1] for i in sorted(retrieved_premises)]
            namespaces, ends, opens = get_env(env[decls[i[1]]['filename']][i[1]])
            i[6] = '```lean\n' + simplify_lean_code('\n\n'.join([
                '\n'.join(f'import {i}' for i in precedent_files), 
                '\n\n'.join(new_premises)
                ] + namespaces + opens + [codes[1]] + ends
                )) + '\n```'
            _, name, context, proof_state, interpretation, sketch, output_proofstep, desc = i
            context = context.strip()
            # decl_description.update((base_prompt.format(instruction=des_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"), interpretation))
            proof_state_interpretation.add((
                multi_line_regex.sub('\n\n', base_prompt.format(instruction=itp_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```")), 
                multi_line_regex.sub('\n\n', interpretation.strip())
                ))
            proof_step_drafting.add((
                multi_line_regex.sub('\n\n', base_prompt.format(instruction=dft_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```")), 
                multi_line_regex.sub('\n\n', '- Overall Planning:\n\n' + sketch.strip())
                ))
            implement = imp_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"
            if retrieved_premises:
                implement += "\n\n- Retrieved Lemmas: \n```lean\n" + simplify_lean_code('\n\n'.join(retrieved_premises)) + "\n```"
            proof_step_implementation.add((
                multi_line_regex.sub('\n\n', base_prompt.format(instruction=implement)), 
                multi_line_regex.sub('\n\n', output_proofstep.strip())
                ))
            # input_dataset.append({
            #     "name": name, 
            #     "context": context, 
            #     "proof_state": proof_state, 
            #     "proof_state_interpretation": interpretation, 
            #     "proofstep_sketch": sketch, 
            #     "proofstep": output_proofstep, 
            #     "decl_description": desc
            # })
            total_len += 1
            tbar.update()
    with open('premises_proof_state_interpretation.json', 'w') as f:
        json.dump([list(i) for i in proof_state_interpretation], f, indent=4, ensure_ascii=False)
    with open('premises_proof_step_drafting.json', 'w') as f:
        json.dump([list(i) for i in proof_step_drafting], f, indent=4, ensure_ascii=False)
    with open('premises_proof_step_implementation.json', 'w') as f:
        json.dump([list(i) for i in proof_step_implementation], f, indent=4, ensure_ascii=False)

def desc():
    # selected_path = ['data', 'algebra', 'analysis', 'field_theory', 'geometry', 'linear_algebra', 'init.data', 'logic', 'number_theory', 'topology.instances', 'order.filter', 'order.well_founded']
    total_len = 0
    decl_description = set()
    loaded_decls = set()
    # manager = Manager()
    # result_queue = manager.Queue()
    with open('description_temp.json') as f:
        desp = json.load(f)
    with open('export_db.json', 'r') as f:
        decls = json.load(f)
    for name, desc in tqdm(list(desp.items())):
        loaded_decls.add(name)
        decl_description.add((
            multi_line_regex.sub('\n\n', base_prompt.format(instruction=des_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{decls[name]['source']}\n```")), multi_line_regex.sub('\n\n', desc.strip())
            ))
    with open('premises_decl_description.json.json', 'w') as f:
        json.dump([list(i) for i in decl_description], f, indent=4, ensure_ascii=False)

def long_main():
    CONTEXT_LEN = 4096
    # selected_path = ['data', 'algebra', 'analysis', 'field_theory', 'geometry', 'linear_algebra', 'init.data', 'logic', 'number_theory', 'topology.instances', 'order.filter', 'order.well_founded']
    total_len = 0
    # decl_description = set()
    long_datasets = set()
    short_datasets = set()
    # manager = Manager()
    # result_queue = manager.Queue()
    with open('description_temp.json') as f:
        desp = json.load(f)
    with open('export_db.json', 'r') as f:
        decls = json.load(f)
    file_graph = nx.read_gexf("import.gexf")
    with open("dependency_graph.pickle", "rb") as f:
        decl_graph = pickle.load(f)
    topological_order = list(nx.topological_sort(decl_graph))
    topological_order.reverse()
    with open('open_env.json') as f:
        env = json.load(f)
    init_end = json.loads(open('decl_init_end.json').read())
    with jsonlines.open('multilevel_proofstep.jsonl') as reader, trange(0) as tbar:
        for i in reader.iter():
            # if not any(decls[i[1]]['filename'].startswith(j) for j in selected_path):
            #     continue
            codes = [i for i in re.findall(r'```Lean3(.*?)```', i[6], re.DOTALL)]
            codes[1] = remove_line_numbers(codes[1]).strip('\n')
            if not len(codes) == 2 or re.sub(r'\s', '', codes[0]) in ['bysorry', 'beginsorryend', 'beginsorry,end']:
                continue
            i.append(desp[i[1]])
            precedent_files = []
            try:
                for pf in file_graph.successors('mathlib:'+decls[i[1]]['filename']):
                    if pf.startswith('mathlib:'):
                        precedent_files.append(pf[len('mathlib:') : ])
            except:
                pass
            new_premises = []
            retrieved_premises = []
            _, proof_step = compute_replacement(remove_line_numbers(i[2]).strip('\n'), codes[1])
            if not proof_step:
                continue
            for premise in [i for i in decl_graph.successors(i[1])]:
                if decls[premise]['filename'] in env and premise in env[decls[premise]['filename']]:
                    if 'mathlib' not in decls[premise]['local_filename'] or premise.split('.')[-1] not in proof_step:
                        continue
                    namespaces, ends, opens = get_env(env[decls[premise]['filename']][premise])
                    if random.random() > 0.5:
                        if decls[premise]['kind'] == 'theorem' and premise in init_end:
                            new_premises.append((topological_order.index(premise),'\n\n'.join(namespaces + opens + [init_end[premise][0]] + ends)))
                        else:
                            new_premises.append((topological_order.index(premise),'\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
                    else:
                        if decls[premise]['kind'] == 'theorem' and premise in init_end:
                            if random.random() > 0.8:
                                retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
                            else:
                                retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [init_end[premise][0]] + ends)))
                        else:
                            retrieved_premises.append((topological_order.index(premise), '\n\n'.join(namespaces + opens + [decls[premise]['source']] + ends)))
            new_premises = [i[1] for i in sorted(new_premises)]
            retrieved_premises = [i[1] for i in sorted(retrieved_premises)]
            namespaces, ends, opens = get_env(env[decls[i[1]]['filename']][i[1]])
            i[6] = '```lean\n' + simplify_lean_code('\n\n'.join([
                '\n'.join(f'import {i}' for i in precedent_files), 
                '\n\n'.join(new_premises)
                ] + namespaces + opens + [codes[1]] + ends
                )) + '\n```'
            _, name, context, proof_state, interpretation, sketch, output_proofstep, desc = i
            context = context.strip()
            # decl_description.update((base_prompt.format(instruction=des_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"), interpretation))
            proof_state_interpretation_item = (multi_line_regex.sub('\n\n', base_prompt.format(instruction=itp_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```")), multi_line_regex.sub('\n\n', interpretation.strip()))
            proof_step_drafting_item = (multi_line_regex.sub('\n\n', base_prompt.format(instruction=dft_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```")), multi_line_regex.sub('\n\n', '- Overall Planning:\n\n' + sketch.strip()))
            implement = imp_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"
            total_prompt = total_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"
            wo_completion_prompt = wo_completion_instruction + f"\n- Declaration Name: {name}\n\n- Context: \n```lean\n{context}\n```\n- Proof State: \n```\n{proof_state}\n```"
            if retrieved_premises:
                retrievals = "\n\n- Retrieved Lemmas: \n```lean\n" + simplify_lean_code('\n\n'.join(retrieved_premises)) + "\n```"
                implement += retrievals
                total_prompt += retrievals
            proof_step_implementation_item = (multi_line_regex.sub('\n\n', base_prompt.format(instruction=implement)), multi_line_regex.sub('\n\n', output_proofstep.strip()))
            total_response = f'''- Translation of Proof States to Natural Language:

{interpretation.strip()}

- Overall Planning:

{sketch.strip()}

- Completion in Lean 3:

{output_proofstep.strip()}'''
            wo_completion_response = f'''- Translation of Proof States to Natural Language:

{interpretation.strip()}

- Overall Planning:

{sketch.strip()}'''
            total_item = (multi_line_regex.sub('\n\n', base_prompt.format(instruction=total_prompt)), multi_line_regex.sub('\n\n', total_response))
            wo_completion_item = (multi_line_regex.sub('\n\n', base_prompt.format(instruction=wo_completion_prompt)), multi_line_regex.sub('\n\n', wo_completion_response))
            if len(tokenizer('\n'.join(total_item))['input_ids']) > CONTEXT_LEN:
                if len(tokenizer('\n'.join(proof_step_implementation_item))['input_ids']) > CONTEXT_LEN:
                    continue
                if len(tokenizer('\n'.join(wo_completion_item))['input_ids']) > CONTEXT_LEN:
                    if len(tokenizer('\n'.join(proof_state_interpretation_item))['input_ids']) > CONTEXT_LEN \
                    or len(tokenizer('\n'.join(proof_step_drafting_item))['input_ids']) > CONTEXT_LEN:
                        continue
                    long_datasets.update((proof_state_interpretation_item, proof_step_drafting_item, proof_step_implementation_item))
                else:
                    long_datasets.update((wo_completion_item, proof_step_implementation_item))
            else:
                short_datasets.add(total_item)
            total_len += 1
            tbar.update()
    print('short_datasets', len(short_datasets))
    print('long_datasets', len(long_datasets))
    with open(f'short_{CONTEXT_LEN}_dataset.json', 'w') as f:
        json.dump([list(i) for i in short_datasets], f, indent=4, ensure_ascii=False)
    with open(f'long_{CONTEXT_LEN}_dataset.json', 'w') as f:
        json.dump([list(i) for i in long_datasets], f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    # main()
    long_main()
    # desc()
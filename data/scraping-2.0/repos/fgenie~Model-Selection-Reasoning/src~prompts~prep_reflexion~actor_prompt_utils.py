# import pandas as pd
# import yaml
# from itertools import product
# from typing import Sequence, Mapping
# from fire import Fire
import re
# import jsonlines as jsl
# from functools import partial
# from pathlib import Path
# from tqdm import tqdm
# import openai
# from tenacity import wait_chain, wait_fixed, retry

# KEY = open('/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/prompts/prep_reflexion/openai_key.txt').read().strip()
# openai.api_key = KEY # set key

'''
read: 
- 2_good_examples_from_*xlsx
- 3_model_select_fs_prompt.yaml
- 3_reflect_fs_prompt.yaml
out:
several
- model select prompts (manual fill-in still left)
- reflection prompts (manual fill-in left)

--> need to manually fill Hints: and Reflection: parts 
--> See those work properly on several questions

'''

# FS_D = pd.read_excel('2_good_examples_from_gsm_test.xlsx', sheet_name=None)
# SELECT_YML = yaml.full_load(open('3_actor_model_select_prompts_plain.yaml'))
# SEL_TMP = SELECT_YML['prompt_template']
# SEL_REF_EXS = SELECT_YML['reflection_exs']
# SEL_ACT_EXS = SELECT_YML['action_exs']

# REFLECT_YML = yaml.full_load(open('3_actor_reflect_prompt_plain.yaml'))
# REF_TMP = REFLECT_YML['prompt_template']
# REF_EXS = REFLECT_YML['reflection_exs']
        




class PromptStr(str): # str class with some utility methods
    def __init__(self, template_str):
        super().__init__()
        self += template_str  

    def sub(self, 
            placeholder_name:str, 
            tobe:str):
        return PromptStr(self.replace(f"[{placeholder_name}]", str(tobe)))

    def get_placeholder_names(self) -> list:
        return re.findall(r"\[(.*?)\]", self)


# def a_fewshot_prep(datarow, 
#                    mode:str='select/reflect')->PromptStr:
#     '''
#     this function works same for SELECTION / REFLECTION (they share the format)
#     (SEL_R_EXS, REF_EXS)

#     datarow: 
#     index	question	ans	correct_pred	wrong_pred	correct_sol	correct_model	wrong_sol	wrong_model

#     '''
#     assert mode in ['select_reflect', 'select_act', 'reflect']
#     if mode == 'select_reflect':
#         shot = PromptStr(SEL_REF_EXS)
#     elif mode == 'select_act':
#         shot = PromptStr(SEL_ACT_EXS)
#     else: # mode == reflect
#         shot = PromptStr(REF_EXS)
    
#     for tagname in shot.get_placeholder_names():
#         key = tagname.lower()
#         if key in datarow.keys():
#             shot = shot.sub(tagname, datarow[key])

#     return shot

# def get_fewshots_string(datarows:Sequence[Mapping], 
#                         mode:str='')->PromptStr:
#     assert mode in ['select_reflect', 'select_act', 'reflect']
#     return PromptStr(
#         '\n\n'.join([a_fewshot_prep(datarow, mode) for datarow in datarows])
#         )


# def fill_selection(reflection_exs:str='', 
#                    action_exs:str='', 
#                    question:str='')->PromptStr:
#     '''
#     reflection_fs and action_fs
#     + question of interest
#     '''
#     # if actions_fs: # process SEL_TMP
#     prompt = PromptStr(SEL_TMP)
#     prompt = prompt.sub('REFLECTION_EXS', reflection_exs)
#     prompt = prompt.sub('ACTION_EXS', action_exs)
#     prompt = prompt.sub('QUESTION', question)

#     # print(prompt.get_placeholder_names())
#     return prompt

# def fill_reflection(reflection_exs:str='',
#                     datarow:str='')->PromptStr:
#     '''
#     reflection_fs and
#     + data to reflect (datarow)
#     '''
#     prompt = PromptStr(REF_TMP)
#     prompt = prompt.sub('REFLECTION_EXS', reflection_exs)
#     for ph in prompt.get_placeholder_names():
#         key = ph.lower()
#         if key in datarow.keys():
#             prompt = prompt.sub(ph, datarow[key])
#     return prompt
        

# def main(outdir='3_prompts_for_manual_fill'):
#     models = ['cot', 'pal', 'p2c']
    
#     outdir = Path(outdir)
#     if not outdir.exists():
#         outdir.mkdir(parents=True, exist_ok=True)
#     for idx in range(3): # try three compositions from the examples
#         datarows = []
#         # get datarows: all 6 model switching cases 
#         for (wm, cm) in product(models, models):
#             if wm == cm:
#                 continue
#             sheetname = f"{wm}_wrong_{cm}_correct"
#             df = FS_D[sheetname]
#             row = df.iloc[idx]
#             # print(wm, cm, idx, row)
#             datarows.append(row)
#         sel_ref_exs = get_fewshots_string(datarows, mode='select_reflect')
#         sel_act_exs = get_fewshots_string(datarows, mode='select_act')
#         ref_exs = get_fewshots_string(datarows, mode='reflect') 
#         # save almost-done prompt
#         s_prompt_temp = fill_selection(reflection_exs=sel_ref_exs, 
#                                        action_exs=sel_act_exs, 
#                                        question='[QUESTION]',
#                                        hint = '[HINT]')
#         r_prompt_temp = fill_reflection(reflection_exs=ref_exs, datarow={})
        
#         with open(outdir/f"reflection_prompt_{idx}.txt", 'w') as rf, open(outdir/f'selection_prompt.txt_{idx}', 'w') as sf:
#             sf.write(s_prompt_temp)
#             rf.write(r_prompt_temp)
#         print(f'saved to {str(outdir)}')


          
            
# def preptest(outdir='3_prompts_for_manual_fill/test',
#              select_prompt_f:str = '3_prompts_for_manual_fill/selection_prompt_0_1.txt',
#              reflect_prompt_f:str = '3_prompts_for_manual_fill/reflection_prompt_0_1.txt'):
#     '''
#     check whether the prompts works as expected
#     expected:
#         1. selection prompt generates:
#             - hint that hints about correct model
#             - selects correct model
#         2. reflection prompt generates:
#             - reflection that pincets what's gone wrong
#             - hints that hints the model selection
#     '''
#     outdir = Path(outdir)
#     if not outdir.exists():
#         outdir.mkdir(parents=True, exist_ok=True)
#     models = ['cot', 'pal', 'p2c']
#     r_tmp = PromptStr(open(reflect_prompt_f).read())
#     s_a0shot = PromptStr(open(select_prompt_f).read())
#     # s_a6shot = PromptStr(open(outdir.parent / 'selection_prompt_0_1_action_fewshots.txt').read()) # I don't think this will work properly (do not want action examples affect the inference)
#     for w, c in product(models, models):
#         if w==c:
#             continue
#         sheet_name = f"{w}_wrong_{c}_correct"
#         df = FS_D[sheet_name]
#         df = df[3:]
#         # complete prompts 
#         prompt_records = [] 
#         for i, row in df.iterrows():
#             s_prompt_0 = s_a0shot # highy suspected...
#             # s_prompt_fs = s_a6shot
#             r_prompt = r_tmp
#             fill = partial(fill_placeholders, datarow=row)
#             # s_prompt_0, s_prompt_fs, r_prompt = map(fill,[s_prompt_0, s_prompt_fs, r_prompt])
#             s_prompt_0, r_prompt = map(fill,[s_prompt_0, r_prompt])
#             # obj = {'selectprompt_0': s_prompt_0, 'selectprompt_fs': s_prompt_fs, 'reflectprompt': r_prompt}
#             obj = {'selectprompt': s_prompt_0, 'reflectprompt': r_prompt}
#             prompt_records.append(obj)
#             # check whether the prompt contains the question
#             assert row.question in s_prompt_0
#             # assert row.question in s_prompt_fs
#             assert row.question in r_prompt         
#         test_data = df.to_dict(orient='records')

#         outdir_ = outdir/f"{Path(select_prompt_f).stem}_{Path(reflect_prompt_f).stem}"
#         if not outdir_.exists():
#             outdir_.mkdir(parents=True, exist_ok=True)
#         with jsl.open(outdir_/f'{sheet_name}_prompt.jsonl', 'w') as writer, jsl.open(outdir_/f'{sheet_name}_data.jsonl', 'w') as writer2:
#             writer.write_all(prompt_records)
#             writer2.write_all(test_data)
#         print(f'saved to {str(outdir)}')

        
# def fill_placeholders(prompt:PromptStr, datarow:dict)->PromptStr:
#     for ph in prompt.get_placeholder_names():
#         key = ph.lower()
#         if key in datarow.keys():
#             prompt = prompt.sub(ph, datarow[key])
#     return prompt  
    
# def test(outroot='3_prompts_for_manual_fill/test', 
#          selection_prompt_f:str='3_prompts_for_manual_fill/selection_prompt_0_1_nobiassys.txt',
#          reflection_prompt_f:str='3_prompts_for_manual_fill/reflection_prompt_0_1.txt',
#          model = 'gpt-3.5-turbo-16k'):
#     outdir = Path(outroot) / f"{Path(selection_prompt_f).stem}_{Path(reflection_prompt_f).stem}"
#     print(outdir)
#     preptest(outdir = outroot, 
#              select_prompt_f=selection_prompt_f, 
#              reflect_prompt_f=reflection_prompt_f)
#     for jslf in tqdm(list(outdir.glob("*prompt.jsonl"))):
#         rescsv = jslf.parent/f"{jslf.stem}_results.jsonl"
#         if rescsv.exists():
#             print(f'skipping {jslf.name} ({rescsv.name} exists)')
#             continue
#         # wrongmodel = jslf.stem.split('_')[0]
#         correctmodel = jslf.stem.split('_')[2]
#         df = pd.DataFrame(jsl.open(jslf))
#         query_results = []
#         for i, row in tqdm(df.iterrows(), total=len(df), desc = jslf.name):
#             rprompt = row.reflectprompt
#             reflection_hint, rtoks_d = query_llm(msgs=[{'role':'user', 'content': rprompt}], model = model, stop = 'Trial Method:')
#             r, h = parse_reflectionhint(reflection_hint)
#             sprompt = PromptStr(row.selectprompt).sub('HINT', h)
#             # print(sprompt)
#             rawselect, stoks_d = query_llm(msgs=[{'role':'user', 'content': sprompt}], model = model, stop='Solution Steps:')
#             select = parse_selection(rawselect)
#             # print(rawselect)
#             # print(select)
#             # print()
#             # print(f'gt: {correctmodel}, select: {select}')
#             obj = {'reflect': r, 
#                    'hint': h,
#                    'select': select, 
#                    'raw_reflect_hint': reflection_hint, 
#                    'raw_select': rawselect}
#             obj.update({f'reflect_{k}': v for k,v in rtoks_d.items()})
#             obj.update({f'select_{k}': v for k,v in stoks_d.items()})
#             query_results.append(obj)
#         results_df = pd.DataFrame(query_results)
#         hits = results_df.select == correctmodel
#         with open(outdir/f'selection_accuracy.txt', 'a') as f:
#             print(f'jslf:{jslf.name}\n\tselection accuracy: {int(hits.sum())}/{len(hits)}={hits.mean()*100:.1f}%', file=f)
#             print(f"\t{results_df.select.value_counts().to_dict()}", file=f)
#         results_records = results_df.to_dict(orient='records')
#         resjsl = outdir/f'{jslf.stem}_results.jsonl'
#         with jsl.open(resjsl, 'w') as writer:
#             writer.write_all(results_records)
#         print('wrote:\n\t', str(resjsl))
            
# @retry(wait=wait_chain(*[wait_fixed(3) for i in range(4)]))
# def query_llm(msgs:list=None, 
#               stop:str='', 
#               max_tokens:int=200,
#               temperature:float=0.,
#               model='gpt-3.5-turbo-16k',
#               )->str:
#     response = openai.ChatCompletion.create(
#         messages = msgs,
#         stop = stop,
#         max_tokens = max_tokens,
#         temperature = temperature,
#         model = model, 
#     )
#     completion = response['choices'][0]['message']['content']
#     usage_dict = response['usage']
#     return completion, usage_dict
    
# def parse_reflectionhint(rawout:str)->tuple:
#     try: 
#         reflect, hint = rawout.split("Hint: ")
#         reflect = reflect.replace('Reflection: ', '').strip()
#         hint = hint.strip()
#     except:
#         reflect, hint = 'failed parsing', rawout
#     return reflect, hint
# def parse_selection(rawout:str)->str:
#     select = rawout.replace("Promising Method: ", '').strip()
#     for candid in ['cot', 'pal', 'p2c']:
#         if candid in select:
#             select= candid
#             break
#     return select

# if __name__ == '__main__':
#     # Fire(main)
#     # Fire(preptest)
#     Fire(test)
#     '''
#     python 3_actor_prompt_test.py --selection_prompt_f 3_prompts_for_manual_fill/selection_prompt_0_1_nobiassys.txt --reflection_prompt_f 3_prompts_for_manual_fill/reflection_prompt_0_1_nobiassys.txt && python 3_actor_prompt_test.py --selection_prompt_f 3_prompts_for_manual_fill/selection_prompt_0_1_nobiassys1.txt --reflection_prompt_f 3_prompts_for_manual_fill/reflection_prompt_0_1_nobiassys1.txt
#     '''

        
        
    
    
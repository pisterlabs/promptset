import datetime
import pdb
from random import random
import re
import time
import openai
from pyparsing import original_text_for
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from datasets import load_metric
from rl4lms.envs.text_generation.metric import BaseMetric
from tqdm import tqdm
import copy
import rouge
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from nltk import sent_tokenize



def extract_keywords(results):
    """
    keyword_list: a list of entities
    keyword_indice_list: a list of indices
    """
    keyword_list = []
    keyword_indice_list = []
    cur_token = ""
    cur_indice = []
    for i, token in enumerate(results):
        if i != 0 and token["start"] == results[i - 1]["end"]:
            if '##' in token["word"]:
                cur_token += token["word"][2:]
            else:
                cur_token += token["word"]
            cur_indice.append(token["index"])
        else:
            if token["score"] > 0.5:
                if token["entity"][0] == "B":
                    if cur_token != "":
                        keyword_list.append(cur_token)
                        keyword_indice_list.append(cur_indice)
                    cur_token = token["word"]
                    cur_indice = [token["index"]]
                else:
                    if i == 0:
                        # error
                        continue
                    elif token["start"] == results[i - 1]["end"] + 1:
                        cur_token += " " + token["word"]
                        cur_indice.append(token["index"])
                    else:
                        # error
                        continue
    keyword_list.append(cur_token)
    keyword_indice_list.append(cur_indice)
    return keyword_list, keyword_indice_list


class GPTROUGEMetric(BaseMetric):
    def __init__(self, use_all_keywords=True,gpt_engine='',openai_api='',kw_model_device=3) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self.openai_api=openai_api
        openai.api_key = openai_api #os.getenv("OPENAI_API_KEY")
        self.use_all_keywords = use_all_keywords
        self._use_single_ref=True
        self.engine_name=gpt_engine
        print('Use model: %s'%(gpt_engine))
        ner_tokenizer = AutoTokenizer.from_pretrained(
            "dslim/bert-large-NER",
            cache_dir="/mnt/default/data/pretrained_models" ,
        )
        ner_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-large-NER",
            cache_dir="/mnt/default/data/pretrained_models",
        )
        self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=kw_model_device)
        
    def generate(self,input_prompt, temperature=0, max_tokens=100,partition_id=None):
        result = None
        retry_interval_exp=1
        openai.api_key = self.openai_api #os.getenv("OPENAI_API_KEY")
        while True:
            try:
                if not partition_id:
                    # Requests will be routed in round robin by default.
                    partition_id = f"sumscience-{datetime.now()}"
                # response=gpt_model.complete(prompt=input_prompt,temperature=temperature,max_tokens=max_tokens)
                response = openai.Completion.create(
                    engine=self.engine_name,
                    # engine="text-davinci-003",
                    prompt=input_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=1,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0.0,
                    stop=None,
                    headers={"partition-id": partition_id},
                )
                result = [
                    response["choices"][i]["text"].replace("\n", "").replace(" .", ".").strip()
                    for i in range(len(input_prompt))
                ]
                return result
            except Exception as e:
                # NOTE: openai.error.RateLimitError: Requests to the
                # Deployments_Completion Operation under OpenAI API have
                # exceeded rate limit of your current OpenAI S0 pricing tier.
                # Please retry after 7 seconds. Please contact Azure support
                # service if you would like to further increase the default rate
                # limit.
                if isinstance(e, openai.error.APIConnectionError):
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2**retry_interval_exp)))
                    retry_interval_exp += 1
                    print('apiconnection error')
                elif isinstance(e,openai.error.RateLimitError):
                    # error = {"type": type(e).__name__, "message": str(e)}
                    # print(error)
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                    retry_interval_exp += 1
                else:
                    # NOTE: openai.error.InvalidRequestError: The response was
                    # filtered due to the prompt triggering Azure OpenAI’s
                    # content management policy.
                    error = {"type": type(e).__name__, "message": str(e)}
                    print(error)
                    return
    def build_prompt_for_correction_instruction(
            self,
            all_inputs,
            instructions
    ):
        all_prompts = []
        orig_summ=[]
        orig_doc=[]
        new_summ_mapping={}
        for i in range(len(all_inputs)):
            summ_doc = re.findall(r'Summary: (?P<summ>.*?)\nDocument: (?P<doc>.*?)\n Instruction',all_inputs[i])
            summ = summ_doc[0][0].strip()
            doc=summ_doc[0][1].strip()
            orig_summ.append(summ)
            orig_doc.append(doc)
            if instructions[i]=='':
                continue 
            prompt = "Document: %s \nSummary: %s \nInstructions: %s \nEdit the summary only following the instructions and only output the corrected summary.\nNew summary: " % (
                doc,
                summ,
                instructions[i]
            )
            new_summ_mapping[i]=len(all_prompts)
            all_prompts.append(prompt)
        return all_prompts,orig_summ,orig_doc,new_summ_mapping

    def build_prompt_for_correction(
            self,
            all_inputs,
            # instructions
            keyword_list_plus,keyword_list_minus
        ):
        # keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(instructions)
        all_prompts = []
        orig_summ=[]
        orig_doc=[]
        new_summ_mapping={}
        for i in range(len(all_inputs)):
            summ_doc = re.findall('<summary>(?P<summ>.*)<document>(?P<doc>.*)',all_inputs[i])
            summ = summ_doc[0][0]
            doc=summ_doc[0][1]
            orig_summ.append(summ)
            orig_doc.append(doc)
            if len(keyword_list_plus[i])==0 and len(keyword_list_minus[i]) == 0:
                continue
            prompt = "Summary: %s \n\nDocument: %s \n\nRewrite the summary for the document," % (
                summ,
                doc,
            )
            if len(keyword_list_plus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " add content related to %s," % (" and ".join(keyword_list_plus[i]))
                else:
                    prompt += " add content related to %s," % (random.choice(keyword_list_plus[i]))
            if len(keyword_list_minus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " delete content related to %s." % (" and ".join(keyword_list_minus[i]))
                else:
                    prompt += " delete content related to %s." % (random.choice(keyword_list_minus[i]))
            prompt += "\n\n New summary: "
            new_summ_mapping[i]=len(all_prompts)
            all_prompts.append(prompt)
        return all_prompts,orig_summ,orig_doc,new_summ_mapping
 

    def build_keyword_list_from_output(self,instructions):
        plus_list_all=[]
        minus_list_all=[]
        for instruction in instructions:
            plus_text=re.search(r'\<add\>(.*?)\<remove\>',instruction)
            if plus_text:
                plus_list=plus_text.group(1)
                plus_list=plus_list.split(',')
                plus_list=[kw.strip() for kw in plus_list if len(kw.strip())>0 and ('##' not in kw) ]
            else:
                plus_list=[]

            minus_text=re.search(r'\<remove\>(.*?)(\<remove\>|\<add\>)',instruction+'<remove>')
            if minus_text:
                minus_list=minus_text.group(1)
                minus_list=minus_list.split(',')
                minus_list=[kw.strip() for kw in minus_list if len(kw.strip())>0 and ('##' not in kw) ]
            else:
                minus_list=[]
            plus_list_all.append(list(set(plus_list)))
            minus_list_all.append(list(set(minus_list)))
        return plus_list_all,minus_list_all
    

    def compute_rouge_keywords(self,all_kw_list_plus,all_kw_list_minus,keyword_list_plus,keyword_list_minus):
        instruction_plus_oracle=[', '.join(kw_list) for kw_list in all_kw_list_plus]
        instruction_minus_oracle=[', '.join(kw_list) for kw_list in all_kw_list_minus]
        instruction_plus_pred=[', '.join(kw_list) for kw_list in keyword_list_plus]
        instruction_minus_pred=[', '.join(kw_list) for kw_list in keyword_list_minus]
        metric_results_plus= self._metric.compute(
            predictions=instruction_plus_pred, references=instruction_plus_oracle, use_stemmer=True,use_aggregator=False
        )
        metric_results_minus= self._metric.compute(
            predictions=instruction_minus_pred, references=instruction_minus_oracle, use_stemmer=True,use_aggregator=False
        )
        # instruction_scores_plus=[np.mean([metric_results_plus['rouge1'][i],metric_results_plus['rouge2'][i]])  for i in range(len(instruction_plus_oracle))]
        instruction_scores_plus=[np.mean([metric_results_plus['rouge1'][i]])  for i in range(len(instruction_plus_oracle))]
        for i in range(len(all_kw_list_plus)):
            if len(all_kw_list_plus[i])==0 and len(keyword_list_plus[i])==0:
                instruction_scores_plus[i]=1
        # instruction_scores_minus=[np.mean([metric_results_minus['rouge1'][i],metric_results_minus['rouge2'][i]])  for i in range(len(instruction_minus_oracle))]
        instruction_scores_minus=[np.mean([metric_results_minus['rouge1'][i]])  for i in range(len(instruction_minus_oracle))]
        for i in range(len(all_kw_list_minus)):
            if len(all_kw_list_minus[i])==0 and len(keyword_list_minus[i])==0:
                instruction_scores_minus[i]=1
        return instruction_scores_plus,instruction_scores_minus
    
    def compute_match_f1_instruction(self,all_kw_list_plus,all_kw_list_minus,keyword_list_plus,keyword_list_minus):
        intersection_num=[len(set(all_kw_list_plus[i]).intersection(set(keyword_list_plus[i]))) for i in range(len(keyword_list_plus))]
        recalls_plus=[1 if len(all_kw_list_plus[i])==0 and len(keyword_list_plus[i])==0 else intersection_num[i]/(len(all_kw_list_plus[i])+1e-9)   for i in range(len(intersection_num))]
        precision_plus=[1 if len(all_kw_list_plus[i])==0 and len(keyword_list_plus[i])==0 else intersection_num[i]/(len(keyword_list_plus[i])+1e-9) for i in range(len(intersection_num))]
        f1_plus = [1 if len(all_kw_list_plus[i])==0 and len(keyword_list_plus[i])==0 else 2*recalls_plus[i]*precision_plus[i]/(recalls_plus[i]+precision_plus[i]+1e-9) for i in range(len(intersection_num))]

        intersection_num=[1 if len(all_kw_list_minus[i])==0 and len(keyword_list_minus[i])==0 else len(set(all_kw_list_minus[i]).intersection(set(keyword_list_minus[i]))) for i in range(len(keyword_list_minus))]
        recalls_minus=[1 if len(all_kw_list_minus[i])==0 and len(keyword_list_minus[i])==0 else intersection_num[i]/(len(all_kw_list_minus[i])+1e-9) for i in range(len(intersection_num))]
        precision_minus=[1 if len(all_kw_list_minus[i])==0 and len(keyword_list_minus[i])==0 else intersection_num[i]/(len(keyword_list_minus[i])+1e-9) for i in range(len(intersection_num))]
        f1_minus = [2*recalls_minus[i]*precision_minus[i]/(recalls_minus[i]+precision_minus[i]+1e-9) for i in range(len(intersection_num))]
        return recalls_plus,precision_plus,f1_plus,recalls_minus,precision_minus,f1_minus

    def compute_reward_kw_match(self,predicted_summary_batch,gt_summary_batch):
        all_summ = predicted_summary_batch+gt_summary_batch
        batch_size = len(predicted_summary_batch)
        all_summ_results = self.ner_pipeline(all_summ)
        gen_summ_results = all_summ_results[:batch_size]
        gt_summ_results = all_summ_results[batch_size:2*batch_size]
        assert len(gt_summ_results) == len(gen_summ_results)
        f1=[]
        recall=[]
        precision = []
        for i_batch in range(len(gt_summ_results)):
            keyword_list_gt, _ = extract_keywords(gt_summ_results[i_batch])
            keyword_list_gen, _ = extract_keywords(gen_summ_results[i_batch])
            num_overlap=len(set(keyword_list_gt).intersection(set(keyword_list_gen)))
            r = num_overlap/(len(keyword_list_gt)+1e-9)
            p=num_overlap/(len(keyword_list_gen)+1e-9)
            f=2*r*p/(r+p+1e-9)
            if len(keyword_list_gen)==0 and len(keyword_list_gt)==0:
                r=1
                p=1
                f=1
            recall.append(r)
            precision.append(p)
            f1.append(f)
        return recall,precision,f1
    
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts
        if 'keyword_list_plus' in meta_infos[0]:
            t = 'kw'
            all_kw_list_plus=[meta_infos[i]['keyword_list_plus'] for i in range(len(meta_infos))]
            all_kw_list_minus=[meta_infos[i]['keyword_list_minus'] for i in range(len(meta_infos))]
            #build prompts to generate summaries
            keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(generated_texts)
            prompts,orig_summ,orig_doc,new_summ_mapping= self.build_prompt_for_correction(prompt_texts,keyword_list_plus,keyword_list_minus)
        elif 'instruction' in meta_infos[0]:
            t='instruction'
            instruction_gt = [meta_infos[i]['instruction'] for i in range(len(meta_infos))]
            #build prompts to generate summaries
            prompts,orig_summ,orig_doc,new_summ_mapping=self.build_prompt_for_correction_instruction(prompt_texts,generated_texts)

        # all_kw_list_plus=[meta_infos[i]['keyword_list_plus'] for i in range(len(meta_infos))]
        # all_kw_list_minus=[meta_infos[i]['keyword_list_minus'] for i in range(len(meta_infos))]
        # keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(generated_texts)
        # prompts,orig_summ,new_summ_mapping = self.build_prompt_for_correction(prompt_texts,keyword_list_plus,keyword_list_minus)
        if len(prompts)>16:
            predicted=[]
            batches = [prompts[i:i+16] for i in range(0,len(prompts),16)]
            for b in tqdm(batches):
                predicted.extend(self.generate(b))
        elif len(prompts)==0:
            predicted=[]
        else:
            predicted = self.generate(prompts)
        if t=='instruction':
            predicted=[sent_tokenize(s)[0] for s in predicted]
        all_predictions=[predicted[new_summ_mapping[i]] if i in new_summ_mapping.keys() else orig_summ[i] for i in range(len(prompt_texts))]
        # predicted=all_predictions
        metric_results = self._metric.compute(
            predictions=all_predictions, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/gptsumm_rouge_{rouge_type}"] = (None, rouge_score)
        if t=='kw':
            instruction_scores_plus,instruction_scores_minus=self.compute_rouge_keywords(all_kw_list_plus,all_kw_list_minus,keyword_list_plus,keyword_list_minus)
            metric_dict["lexical/instruction_plus_rouge_1"] = (instruction_scores_plus, np.mean(instruction_scores_plus))
            metric_dict["lexical/instruction_minus_rouge_1"] = (instruction_scores_minus, np.mean(instruction_scores_minus))

            recalls_plus,precision_plus,f1_plus,recalls_minus,precision_minus,f1_minus = self.compute_match_f1_instruction(all_kw_list_plus,all_kw_list_minus,keyword_list_plus,keyword_list_minus)
            metric_dict["lexical/instruction_plus_recall"]=(recalls_plus,np.mean(recalls_plus))
            metric_dict["lexical/instruction_plus_precision"]=(precision_plus,np.mean(precision_plus))
            metric_dict["lexical/instruction_plus_f1"]=(f1_plus,np.mean(f1_plus))
            metric_dict["lexical/instruction_minus_recall"]=(recalls_minus,np.mean(recalls_minus))
            metric_dict["lexical/instruction_minus_precision"]=(precision_minus,np.mean(precision_minus))
            metric_dict["lexical/instruction_minus_f1"]=(f1_minus,np.mean(f1_minus))
        elif t=='instruction':
            metric_results = self._metric.compute(
                predictions=generated_texts, references=instruction_gt, use_stemmer=True
            )
            score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            for rouge_type in score_keys:
                rouge_score = metric_results[rouge_type].mid.fmeasure
                metric_dict[f"lexical/instruction_rouge_{rouge_type}"] = (None, rouge_score)

        recall,precision,f1=self.compute_reward_kw_match(all_predictions,ref_texts)
        metric_dict["lexical/keyword_matching_recall"]=(recall,np.mean(recall))
        metric_dict["lexical/keyword_matching_precision"]=(precision,np.mean(precision))
        metric_dict["lexical/keyword_matching_f1"]=(f1,np.mean(f1))

        metric_dict["gpt_summ"]=(all_predictions,None)
        metric_dict["num_completed"]=(None,len(prompt_texts)-len(prompts))
        return metric_dict
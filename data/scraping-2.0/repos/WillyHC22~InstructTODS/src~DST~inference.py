import pandas as pd
from tqdm import tqdm
from langchain import PromptTemplate

from dst import SLOTS_REVERSE_REMAPPING
from config import CONFIG
from dst_utils import MWOZ_Dataset
from evaluate_utils import unpack_belief_states

import openai

openai.organization = CONFIG["openai_organization"]
openai.api_key= CONFIG["openai_api_key"]


def generate(model, prompt):
    
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        completion = openai.ChatCompletion.create(
          model=model, 
          messages=[{"role": "user", "content": prompt}
                    ],
          temperature=0,
        )
        return completion["choices"][0]["message"]["content"], completion
    else:
        return "Only gpt-3.5 and gpt-4 currently"

def predict(dataset, model, save_path, start_idx, save_every, debug):
    if debug:
        dataset = dataset[:50]
    else:
        dataset = dataset[start_idx:]
    outputs = dataset.copy(deep=True)
    model_used = []
    preds = []
    completion_info = []
    temp_prompts = []
    temp_idx = []
    for idx, row in tqdm(dataset.iterrows()):
        prompt = row["prompt"]
        sample_id = row["id"]
        retry_count = 0
        while True:
            if retry_count > 5:
                print("Retried too many times")
                break
            try:
                pred, completion = generate(model, prompt)
                break
            except:
                retry_count += 1
                continue
        if retry_count > 5:
            break

        model_used.append(model)
        preds.append(pred)
        completion_info.append(completion)
        temp_idx.append(sample_id)

        #TO DELETE LATER
        temp_prompts.append(prompt)
        if idx % save_every == 0:
            temp_save_path = save_path[:-4] + "_latestSave.csv"
            temp_df = pd.DataFrame({"id":temp_idx,
                                    "prompts":temp_prompts,
                                    "preds":preds,
                                    "completion_info":completion_info})
            temp_df.to_csv(temp_save_path)
    
    outputs["model_used"] = model_used
    outputs["preds"] = preds
    outputs["completion_info"] = completion_info
    df = pd.DataFrame(outputs)
    df.to_csv(save_path)
    return df



def predict_correct_bs(results_df, dataset, model, config, save_path, save_every, with_slot_description):
    template = config["PROMPT_TEMPLATES"]["template_with_slots_recorrect"]
    instruction = config["INSTRUCTIONS"]["instruction_with_slots_recorrect_3"]
    prompt_template = PromptTemplate(input_variables= template["input_variables"],
                                     template = template["template"])
    model_used = []
    preds = []
    completion_info = []
    temp_prompts = []
    temp_idx = []

    for idx, row in tqdm(results_df.iterrows()):
        original_preds = row["preds"]
        belief_states = {}
        for slot_value in unpack_belief_states(original_preds, "pred"):
            slot, value = slot_value.split("-")[0], slot_value.split("-")[1]
            if slot in SLOTS_REVERSE_REMAPPING:
                slot = SLOTS_REVERSE_REMAPPING[slot]
            belief_states[slot] = value

        original_prompt = row["prompt"]
        sample_id = row["id"]
        if not with_slot_description:
            slots, context = original_prompt.split("CONTEXT:")[0], original_prompt.split("CONTEXT:")[1]
            slots = slots.split("SLOTS:")[1]
            slots = slots.replace("\n", "")
        else:
            dialogue_id = row["id"]
            #Get the prompt in dataset with same id
            if "description" not in dataset["prompt"][0]:
                raise ValueError("Missing descriptions in the dataset provided, make sure the dataset is built with argument with_slot_description=True")
            original_prompt = dataset.loc[dataset["id"] == dialogue_id]["prompt"].item()
            slots, context = original_prompt.split("CONTEXT:")[0], original_prompt.split("CONTEXT:")[1]
            slots = slots.split("SLOTS:")[1] 
        
        if len(template["input_variables"]) == 4:
            new_prompt = prompt_template.format(instruction=instruction.replace("\n", ""),
                                                slots=slots,
                                                dialogue_context=context.replace("\n", ""),
                                                belief_states=belief_states)
        elif len(template["input_variables"]) == 3:
            instruction = instruction.replace(r"{belief_state}", str(belief_states))
            new_prompt = prompt_template.format(instruction=instruction.replace("\n", ""),
                                                slots=slots,
                                                dialogue_context=context.replace("\n", ""))            

        retry_count = 0
        while True:
            if retry_count > 5:
                print("Retried too many times")
                break
            try:
                retry_count += 1
                pred, completion = generate(model, new_prompt)
                break
            except:
                continue
        if retry_count > 5:
            break
        
        temp_prompts.append(new_prompt)
        model_used.append(model)
        preds.append(pred)
        completion_info.append(completion)
        temp_idx.append(sample_id)

        if idx % save_every == 0:
            temp_save_path = save_path[:-4] + "_latestSave.csv"
            temp_df = pd.DataFrame({"id":temp_idx,
                                    "prompt":temp_prompts,
                                    "correct_preds":preds,
                                    "completion_info":completion_info})
            temp_df.to_csv(temp_save_path)
    

    if len(results_df) == len(temp_prompts):
        results_df["prompt_recorrected"] = temp_prompts

    results_df["model_used"] = model_used
    results_df["correct_preds"] = preds
    results_df["completion_info"] = completion_info

    results_df.to_csv(save_path)
    return results_df
        


if __name__ == "__main__":

    mwoz_path = "/home/willy/instructod/MultiWOZ_2.1/"
    dialog_history_limit = 0 #+1 utterances
    with_slot_description = True
    single_domain_only = False
    with_req_inf_differentiation = False
    with_all_slots = True
    mwoz = MWOZ_Dataset(config=CONFIG, 
                        mwoz_path=mwoz_path,
                        dialog_history_limit=dialog_history_limit,
                        with_slot_description=with_slot_description,
                        with_req_inf_differentiation=with_req_inf_differentiation,
                        single_domain_only=single_domain_only,
                        with_all_slots=with_all_slots)
    dataset = mwoz.dataset

    model = "gpt-3.5-turbo"
    save_every = 25
    start_idx = 2901
    end_idx = "end"
    debug = False
    
    save_path = f"/home/willy/instructod/src/DST/results_single/{model}_{start_idx}-{end_idx}_debug{debug}_singleDomainOnly{single_domain_only}_withSlotDescription{with_slot_description}_withSlotDifferentiation{with_req_inf_differentiation}_withAllSlots{with_all_slots}_dialogHistoryLimit{dialog_history_limit}_prompt3.csv"
    results_df = predict(dataset, model, save_path, start_idx, save_every, debug)


    #------------------------------------------------------
    # print("RUNNING DST RECORRECTION")
    # save_path = "/home/willy/instructod/src/DST/results_single/gpt-4_0-end_recorrectSlotDescFalse_debugFalse_singleDomainOnlyTrue_withSlotDescriptionFalse_withSlotDifferentiationFalse_withAllSlotsTrue_dialogHistoryLimit0_prompt3.csv"
    # load_path = "/home/willy/instructod/src/DST/results_single/gpt-4_0-end_debugFalse_singleDomainOnlyTrue_withSlotDescriptionFalse_withSlotDifferentiationFalse_withAllSlotsTrue_dialogHistoryLimit0_prompt3.csv"
    # previous_results = pd.read_csv(load_path)
    # results_df = predict_correct_bs(results_df=previous_results,
    #                                 dataset=dataset,
    #                                 model=model, 
    #                                 config=CONFIG, 
    #                                 save_path=save_path,
    #                                 save_every=save_every,
    #                                 with_slot_description=with_slot_description)

import os
import json
import pandas as pd

from tqdm import tqdm
from langchain import PromptTemplate

from evaluate_utils import remapping
from dst import SLOTS_DESCRIPTIONS
from config import CONFIG



class PromptConstructor():
    def __init__(self, 
                 config):
        self.config = config
        self.instructions = config["INSTRUCTIONS"]
        self.prompt_templates = config["PROMPT_TEMPLATES"]
        
    def _get_slots_from_domains(self, domains, with_slot_description, with_req_inf_differentiation, with_all_slots):
        # slot_description = self.config["slot_descrpition"]
        if with_all_slots:
            domains = "all"
        
        if with_slot_description:
            with_req_inf_differentiation = False #Slot description is the discriminator

        if domains == "all":
            if with_req_inf_differentiation:
                req_slots = ", ".join(self.config["multiwoz21"]["all_requestable_slots"])
                inf_slots = ", ".join(self.config["multiwoz21"]["all_informable_slots"])
            else:
                slots = set(self.config["multiwoz21"]["all_requestable_slots"] + 
                            self.config["multiwoz21"]["all_informable_slots"])
                slots = ", ".join(slots)
        elif not isinstance(domains, list):
            raise ValueError("""Provided domain should be either 'all' or list of valid domain names:
                                - for multiwoz2.1 and 2.4: taxi, restaurant, hotel, train, attraction 
                                - for SGD: To-do""")
        else:
            req_slots = ""
            inf_slots = ""
            domain_req_slots = []
            domain_inf_slots = []
            for domain in domains:
                domain_req_slots += self.config["multiwoz21"]["requestable_slots"][domain]
                domain_inf_slots += self.config["multiwoz21"]["informable_slots"][domain]
            if with_req_inf_differentiation:
                domain_req_slots = set(domain_req_slots)
                domain_inf_slots = set(domain_inf_slots)
                req_slots += ", ".join(domain_req_slots)
                inf_slots += ", ".join(domain_inf_slots)
            else:
                slots = set(domain_req_slots + domain_inf_slots)
                slots = ", ".join(slots)

        if with_req_inf_differentiation:
            slots_info = f"Requestable slots: {req_slots}\nInformable slots: {inf_slots}"
        else:
            slots_info = f"{slots}"

        if with_slot_description:
            slots = slots.split(", ")
            slots_info = ""
            for slot in slots:
                if slot not in self.config["multiwoz21"]["all_informable_slots"]:
                    continue
                slots_info += f"name: {slot}, description: {SLOTS_DESCRIPTIONS[slot]}\n"
            slots_info = slots_info[:-2]
        
        return slots_info
    
    
    def _build_prompt(self, mode="", dialogue_context="", ontology="", slots="", dialogue_acts="", belief_states=""):
        prompt = ""
        if mode == "dst":
            instruction = self.instructions["instruction_with_slots"]
            template_variables = self.prompt_templates["template_with_slots"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                     slots=slots,
                                     dialogue_context=dialogue_context)
            
        elif mode == "dst_recorrect":
            instruction = self.instructions["instruction_with_slots_recorrect"]
            template = self.prompt_templates["template_with_slots_recorrect"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])            
            prompt = template.format(instruction=instruction,
                                    slots=slots,
                                    dialogue_context=dialogue_context,
                                    belief_states=belief_states)
            
        elif mode == "database_query":
            instruction = self.instructions["instruction_query_database"]
            template = self.prompt_templates["template_query_database"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                    belief_states=belief_states)
            
        elif mode == "response_generation":
            instruction = self.instructions["instruction_response_generation"]
            template = self.prompt_templates["template_response_generation"]
            template = PromptTemplate(input_variables = template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                    dialogue_acts=dialogue_acts,
                                    dialogue_context=dialogue_context)
        elif mode == "dst_extracted_ontology":
            pass

        else:
            raise ValueError("'mode' should be one of: [dst, dst_recorrect, database_query, response_generation]")
        
        return prompt


class MWOZ_Dataset(PromptConstructor):
    def __init__(self,
                 config,
                 mwoz_path,
                 dialog_history_limit,
                 with_slot_description,
                 with_req_inf_differentiation,
                 single_domain_only,
                 with_all_slots):
        PromptConstructor.__init__(self, config)
        self.dataset = {"id":[],
                        "dialogue_id":[],
                        "dialogue_context":[],
                        "turn":[],
                        "prompt":[],
                        "domains":[],
                        "gold_turn_bs":[],
                        "gold_bs":[],
                        "gold_act":[],
                        "gold_response":[],
                        "gold_database_result":[],
                        }
        self.all_data, self.testfiles = self._get_mwoz_data(mwoz_path)
        self.idx = 0
        self.dialog_history_limit = dialog_history_limit
        self.single_domain_only = single_domain_only
        self.with_slot_description = with_slot_description
        self.with_req_inf_differentiation = with_req_inf_differentiation
        self.with_all_slots = with_all_slots

        print("Processing mwoz...")
        for sample in tqdm(self.all_data):
            if sample in self.testfiles:
                dialogue_log = self.all_data[sample]["log"]
                self._process_dialogue_log(sample=sample,
                                           dialogue_log=dialogue_log)

        self.dataset = pd.DataFrame(self.dataset)
        if single_domain_only:
            for index, row in tqdm(self.dataset.iterrows()):
                if len(row["domains"]) != 1:
                    self.dataset.drop(index, inplace=True)

    def _get_mwoz_data(self, mwoz_path):
        data_path = os.path.join(mwoz_path, "data.json")
        testListFile_path = os.path.join(mwoz_path, "testListFile.txt")

        with open(data_path, "r") as f:
            all_data = json.load(f)
            
        with open(testListFile_path, "r") as f:
            testfiles = f.read()
        testfiles = testfiles.split("\n")
        return all_data, testfiles
    
    def _process_dialogue_log(self, sample, dialogue_log):

        dialog_history_memory = []
        dialog_history = ""
        domains = self._get_domains_from_log(dialogue_log)
        slots = self._get_slots_from_domains(domains, 
                                             self.with_slot_description,
                                             self.with_req_inf_differentiation,
                                             self.with_all_slots) # or all

        for turn_nb, turn in enumerate(dialogue_log):

            if turn_nb % 2 == 0:
                speaker = "USER"
            else:
                speaker = "SYSTEM"

            utterance = f"""{speaker}: {turn["text"]}\n"""
            dialogue_context = dialog_history + utterance
            dialog_act = turn["dialog_act"]
            prompt = self._build_prompt(mode="dst",
                                        slots=slots,
                                        dialogue_context=dialogue_context) 


            if self.dialog_history_limit != 0:
                if self.dialog_history_limit == -1:
                    self.dialog_history_limit = len(dialogue_log)

                if len(dialog_history_memory) >= self.dialog_history_limit:
                    dialog_history_memory.pop(0)
                dialog_history_memory.append(utterance)
                dialog_history = "".join(dialog_history_memory)

            metadata = turn["metadata"]
            bspn_dict = {}
            if metadata:
                for domain in metadata:
                    slot_values = metadata[domain]["semi"]
                    for slot in slot_values:
                        value = slot_values[slot]
                        if value and value not in ["not mentioned", "none"]:
                            if domain in bspn_dict:
                                bspn_dict[domain].append(remapping(slot))
                                bspn_dict[domain].append(remapping(value))
                            else:
                                bspn_dict[domain] = [remapping(slot), remapping(value)]
                bspn = " ".join([f"[{domain}] {' '.join(bspn_dict[domain])}" for domain in bspn_dict])

            self.idx += 1
            if turn_nb % 2 == 0:
                self.dataset["gold_turn_bs"].append(dialog_act)
                self.dataset["dialogue_context"].append(dialogue_context)
                self.dataset["gold_database_result"].append(None) 
                self.dataset["turn"].append(turn_nb//2)
                self.dataset["domains"].append(domains)
                self.dataset["id"].append(self.idx//2)
                self.dataset["dialogue_id"].append(sample)
                self.dataset["prompt"].append(prompt)
            else:
                self.dataset["gold_response"].append(utterance)
                self.dataset["gold_bs"].append(bspn)
                self.dataset["gold_act"].append(dialog_act)


    def _get_domains_from_log(self, dialogue_log):
        domains = []
        all_domains = ["restaurant", "taxi", "hotel", "train", "attraction"]
        for log in dialogue_log:
            for domain_act in log["dialog_act"]:
                domain = domain_act.split("-")[0].lower()
                if domain in all_domains and domain not in domains:
                    domains.append(domain)
        return domains
                

if __name__ == "__main__":
    # mwoz_path = "/home/willy/InstrucTOD/MultiWOZ_2.1/"
    mwoz_path = "/home/willy/instructod/MultiWOZ_2.1/"
    dialog_history_limit = 0
    single_domain_only = False
    with_slot_description = False
    with_req_inf_differentiation = False
    with_all_slots = True
    mwoz = MWOZ_Dataset(CONFIG, 
                        mwoz_path,
                        dialog_history_limit,
                        with_slot_description,
                        with_req_inf_differentiation,
                        single_domain_only,
                        with_all_slots)
    dataset = mwoz.dataset
    for i in range(10):
        for key in dataset:
            print(f"""{key}: {dataset[key][i]}""")
        print("================")

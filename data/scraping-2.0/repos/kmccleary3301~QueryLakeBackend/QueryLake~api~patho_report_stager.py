from ..models.model_manager import LLMEnsemble, num_tokens_from_string
from .llm_model_calls import llm_call_model_synchronous
import csv
from io import BytesIO, StringIO
# from .hashing import random_hash
from .patho_references import TNM_categories_fill_in_refit
# import TNM_categories_fill_in_refit
import re, os, json
import random
import numpy as np
from copy import deepcopy
import textwrap
import itertools
import time
from sqlmodel import Session, select, and_, not_
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# os.chdir(globals()['_dh'][0])

category_explanations = {
    "T": "Primary tumor Anatomic Staging: Clinical and Pathologic",
    "N": "Clinical Anatomic Regional Lymph Node Staging",
    "M": "Distant Metastases: Anatomic Staging (Clinical and Pathologic)"
}

category_table = {
    "T": {
        "Tx": "Primary tumor cannot be assessed",
        "T0": "No evidence of primary tumor",
        "Tis (DCIS)": "Ductal carcinoma in situ",
        "Tis (Paget)": "Paget disease not associated with invasive carcinoma or DCIS",
        "T1mi": "Largest Tumor size <= 1 mm",
        "T1a": "Largest Tumor size > 1 mm but <= 5 mm",
        "T1b": "Largest Tumor size > 5 mm but <= 10 mm",
        "T1c": "Largest Tumor size > 10 mm but <= 20 mm",
        "T2": "Largest Tumor size > 20 mm but <= 50 mm",
        "T3": "Largest Tumor size > 50 mm",
        "T4a": "Largest Tumor with chest wall invasion",
        "T4b": "Largest Tumor with macroscopic skin changes including ulceration and/or satellite skin nodules and/or edema",
        "T4c": "Largest Tumor meets criteria of both T4a and T4b",
        "T4d": "Inflammatory carcinoma" # No references available for this case
    }, 
    "N": {
        "cNx": "Regional nodes cannot be assessed (previously removed)",
        "cN0": "No regional nodal metastases",
        "cN1": "Metastases to movable ipsilateral level I and/or level II axillary nodes",
        "cN1mi": "Micrometastases",
        "cN2a": "Metastases to fixed or matted ipsilateral level I and/or level II axillary nodes",
        "cN2b": "Metastases to ipsilateral internal mammary nodes without axillary metastases",
        "cN3a": "Metastases to ipsilateral level III axillary nodes with or without level I and/or level II axillary metastases",
        "cN3b": "Metastases to ipsilateral internal mammary nodes with level I and/or level II axillary metastases",
        "cN3c": "Metastases to ipsilateral supraclavicular nodes"
    },
    "M": {
        "Mx": "Distant metastases cannot be assessed",
        "M0": "No clinical or imaging evidence of distant metastases",
        "cM0(i+)": "No clinical or imaging evidence of distant metastases, but with tumor cells or deposits measuring <= 0.2 mm detected in circulating blood, bone marrow, or other nonregional nodal tissue in the absence of clinical signs and symptoms of metastases",
        "cM1": "Distant metastases on the basis of clinical or imaging findings",
        "pM1": "Histologically proven distant metastases in solid organs; or, if in nonregional nodes, metastases measuring >0.2 mm"
    }
}


pass_categories_single = """Below is an OCR'd pathology report of a breast cancer case.
Assess the information in the report to decide whether the case described in the
report meets the following categorical description:

context: {category_explanation}

### PRIMARY QUESTION:
Does the below report meet the criteria of the following stage?
<STAGE_TO_ASSES>
Stage       |   Description
{stage_entry}
</STAGE_TO_ASSES>

For reference, here are all known categories:
Stage       |   Description
{all_categories}

Format your response as a json, and include 3 fields within the json. 
The first field for your conclusion (true or false, as a boolean).
Only write true if you are completely confident that it matches the provided description.
If you are uncertain in any way, or the report does not fit the description perfectly and exactly,
fill this field as false.

The second field for citing segments of the report relevant to your conclusion, formatted as an
array of strings (if there is not an applicable category, leave this field as null). 
In the cited_lines field in the example below, the strings are possible examples. These are placeholders.
Replace these with unmodified text from the report that was relevant to your conclusion.

The third field for your reasoning, formatted as a single string. 
The keys in your json should be \"report_meets_description\", \"cited_lines\", and \"reasoning\".


# Report:
<REPORT>
{report}
</REPORT>

# Response:
"""

pass_categories_multiple = """Below is an OCR'd pathology report of a breast cancer case.
Assess the information in the report to decide whether the case described in the
report meets one of the following categorical description:

context: {category_explanation}

Stage       |   Description
{stage_table}

Format your response as a json, and include 3 fields within the json. 
The first field for your conclusion. For this, either enter the stage that is most applicable
from the above table.
If you are at all uncertain in any way about your conclusion, fill this field as 'UNKNOWN'.

The second field for citing segments of the report relevant to your conclusion, formatted as an
array of strings (if there is not an applicable category, leave this field as null). 
In the cited_lines field in the example below, the strings are possible examples. These are placeholders.
Replace these with unmodified text from the report that was relevant to your conclusion.

The third field for your reasoning, formatted as a single string. 
The keys in your json should be \"report_meets_description\", \"cited_lines\", and \"reasoning\".


# Report:
<REPORT>
{report}
</REPORT>

# Response:
"""
SAMPLES = {}
SAMPLES["explicit_stage_statements"] = [line for line in TNM_categories_fill_in_refit.explicit_stage_statement_examples.split("\n") if line.strip() != ""]
SAMPLES["measurement_samples"] = [line for line in TNM_categories_fill_in_refit.t_measurement_statement_examples.split("\n") if line.strip() != ""]
SAMPLES["T"] = {}
SAMPLES["N"] = {}
SAMPLES["M"] = {}
t_lookup_original = TNM_categories_fill_in_refit.t_lookup[0]
# n_lookup_original = TNM_categories_fill_in_completed.n_lookup[0]
# m_lookup_original = TNM_categories_fill_in_completed.m_lookup[0]
# print(TNM_categories_fill_in_refit.t_lookup)
for category_stage, _ in t_lookup_original.items():
    # print(category_stage)
    SAMPLES["T"][category_stage] = {}
    for key, value in t_lookup_original[category_stage].items():
        SAMPLES["T"][category_stage][key] = [line for line in value.split("\n") if line.strip() != ""]
for category_stage, _ in TNM_categories_fill_in_refit.n_lookup.items():
    SAMPLES["N"][category_stage] = {}
    for key, value in TNM_categories_fill_in_refit.n_lookup[category_stage].items():
        SAMPLES["N"][category_stage][key] = [line for line in value.split("\n") if line.strip() != ""]
for category_stage, _ in TNM_categories_fill_in_refit.m_lookup.items():
    SAMPLES["M"][category_stage] = {}
    for key, value in TNM_categories_fill_in_refit.m_lookup[category_stage].items():
        SAMPLES["M"][category_stage][key] = [line for line in value.split("\n") if line.strip() != ""]

for category in ["T", "N", "M"]:
    for category_stage in SAMPLES[category].keys():
        SAMPLES[category][category_stage]["Definition"] = SAMPLES[category][category_stage]["Notes"][0]

with open("reformated_fill_in.json", "w+", encoding="utf-8") as f:
    f.write(json.dumps(SAMPLES, indent=4))
    f.close()

T_CATEGORIES = [category_stage for category_stage, _ in t_lookup_original.items()]
N_CATEGORIES = [category_stage for category_stage, _ in TNM_categories_fill_in_refit.n_lookup.items()]
M_CATEGORIES = [category_stage for category_stage, _ in TNM_categories_fill_in_refit.m_lookup.items()]

T_MEASURMENT_LOOKUP = {
    "Tx": {}, #Primary tumor exists, but cannot be assessed.
    "T0": {}, 
    "Tis (DCIS)": {}, 
    "Tis (Paget)": [],
    "T1mi": {"range": [0, 1, True]},
    "T1a": {"range": [1, 5, True]},
    "T1b": {"range": [5, 10, True]},
    "T1c": {"range": [10, 20, False]},
    "T2": {"range": [20, 50, False]},
    "T3": {"range": [50, 200, False]},
    "T4a": {"range": [50, 200, False]},
    "T4b": {"range": [50, 200, False]},
    "T4c": {"range": [50, 200, False]},
    "T4d": {"range": [50, 200, False]},
}

def shuffle_seed(list_in, seed = None):
    """
    Shuffles a list randomly. 
    If a seed is provided, it does so with said seed.
    """

    if seed is None:
        seed = random.randrange(0, 2**32-1)
    list_shape = np.shape(list_in)
    np.random.seed(seed)
    if len(list_shape) != 1:
        new_list = np.ndarray.flatten(np.array(list_in))
    else:
        new_list = np.array(list_in)
    data_length = len(new_list)
    shuf_order = np.arange(data_length)
    np.random.shuffle(shuf_order)
    shuffled_flat = new_list[shuf_order]
    return np.reshape(shuffled_flat, list_shape)

def generate_measurement(min : float, max : float, use_mm : bool = True, return_single : bool = False, return_both : bool = False) -> str:
    """
    Generate M x M x M measurements with either mm or cm.
    Args are in mm always.
    """
    (m_1, m_2, m_3) = tuple(sorted([random.uniform(min, max), random.uniform(0.1, max), random.uniform(0.1, max)], reverse=True))
    if not use_mm:
        m_1, m_2, m_3 = m_1 / 10, m_2 / 10, m_3 / 10
    output = "%.1f x %.1f x %.1f" % tuple(shuffle_seed([m_1, m_2, m_3]))
    
    if return_both:
        output_2 = "%.1f" % m_1
        if use_mm:
            return output + " mm", output_2 + " mm"
        return output + " cm", output_2 + " cm"

    if return_single:
        output = "%.1f" % m_1
    if use_mm:
        return output + " mm"
    return output + " cm"



def assess_category_single(llm, report, category, category_stage):
    category_explanation = category_explanations[category]
    # print(category, category_stage)
    category_stage_description = category_table[category][category_stage]
    
    # category_explanation = category_explanations[category]
    # print("1,", list(SAMPLES[category].keys()))
    # print("2,", list(SAMPLES[category][category_stage].keys()))
    ref_dictionary = deepcopy(SAMPLES[category][category_stage])
    # print(ref_dictionary)
    ref_dictionary["Sample Sentences"] = ref_dictionary["Sample Sentences"][:min(5, len(ref_dictionary["Sample Sentences"]))]
    if "Negative Sample Sentences" in ref_dictionary:
        ref_dictionary["Negative Sample Sentences"] = ref_dictionary["Negative Sample Sentences"][:min(5, len(ref_dictionary["Negative Sample Sentences"]))]
    category_explanation = "CATEGORY: \"%s\"\n" % (category_stage) + json.dumps(ref_dictionary, indent=4)
    category_stages = list(SAMPLES[category].keys())
    category_stage_descriptions = [SAMPLES[category][key]["Notes"] for key in category_stages]
    table_format = "\n".join(["%-12s|   %s" % (category_stages[i], category_stage_descriptions[i]) for i in range(len(category_stages))])
    
    stage_entry = "%-12s|   %s" % (category_stage, category_stage_description)

    prompt_template = PromptTemplate(input_variables=["report", "category_explanation", "all_categories", "stage_entry"], template=pass_categories_single)
    final_prompt = prompt_template.format(
                                            report=report, 
                                            category_explanation=category_explanation, 
                                            all_categories = table_format,
                                            stage_entry=stage_entry
                                         )
    return final_prompt

def assess_categories_multiple(llm, report, category, category_stages):
    category_explanation = category_explanations[category]
    category_stage_descriptions = [category_table[category][key] for key in category_stages]

    # table_format = "\n".join(["%-12s|   %s" % (category_stages[i], category_stage_descriptions[i]) for i in range(len(category_stages))])
    table_format = ""
    for category_stage in category_stages:
        ref_dictionary = deepcopy(SAMPLES[category][category_stage])
        # print(ref_dictionary)
        ref_dictionary["Sample Sentences"] = ref_dictionary["Sample Sentences"][:min(5, len(ref_dictionary["Sample Sentences"]))]
        if "Negative Sample Sentences" in ref_dictionary:
            ref_dictionary["Negative Sample Sentences"] = ref_dictionary["Negative Sample Sentences"][:min(5, len(ref_dictionary["Negative Sample Sentences"]))]
        category_explanation_tmp = "CATEGORY: \"%s\"\n" % (category_stage) + json.dumps(ref_dictionary, indent=4)
        table_format += category_explanation_tmp + "\n"

    # print("generated table:")
    # print(table_format)
    
    prompt_template = PromptTemplate(input_variables=["report", "category_explanation", "stage_table"], template=pass_categories_multiple)
    final_prompt = prompt_template.format(
                                            report=report, 
                                            category_explanation=category_explanation, 
                                            stage_table=table_format
                                         )
    return final_prompt

def stage_breast_cancer_report(database : Session,
                                username : str, 
                                password_prehash : str,
                                llm_ensemble,
                                report: str,
                                report_title: str,
                                model_choice : str = None):
    def distil_with_single_passes(input_report, 
                                categorical_mapping, 
                                category):
        new_mapping = []

        prompt_tokens, response_tokens = 0, 0

        for i, category_stage in enumerate(categorical_mapping):
            percent_complete = i / (len(categorical_mapping) - 1)
            bar_fill_count = int(20*percent_complete)
            progress_bar = "=" * bar_fill_count + ">" + " "*(20 - bar_fill_count)
            print("Assessing %17s [%s] %3.2f%% %s" % (category_stage, progress_bar, 100*percent_complete, " "*10), end="\r")
            # print(list(category_table[category].keys()))
            # print(category, category_stage)
            final_prompt = assess_category_single(None, input_report, category=category, category_stage=category_stage)
            # final_prompt = assess_category_multiple()
            formatted_chat_history = [
                {"role": "system", "content": "You are a classification assistant that helps classify medical reports. Respond to the following question in JSON format, according to the user's request. Do not include anything else."},
                {"role": "user", "content": final_prompt}
            ]

            # result_data = run_openai_model(formatted_chat_history, openai_model, api_kwargs)
            result_data = llm_call_model_synchronous(database, llm_ensemble, username, password_prehash, formatted_chat_history, model_choice=model_choice)

            # if "openai" in model_choice:
            #     for chat_entry in formatted_chat_history:
            #         prompt_tokens += num_tokens_from_string(chat_entry["content"], model_choice)
            #     prompt_tokens += num_tokens_from_string("{\n\"report_meets_description\": ", model_choice)
            #     response_tokens += num_tokens_from_string(result_data[len("{\n\"report_meets_description\": "):], model_choice)
            

            result_data = re.sub(r"^[^\{]*", "", re.sub(r"[^\}]*$", "", result_data))
            # print("Looking for stage:", category_stage, category_table[category][category_stage])
            # print(result_data)
            try:
                parsed = json.loads(result_data)
                if "report_meets_description" not in parsed:
                    assert False, "description validation not in json"
                if parsed["report_meets_description"]:
                    new_mapping.append(category_stage)
            except:
                # print("Failed to parse JSON")
                new_mapping.append(category_stage)
        return prompt_tokens, response_tokens, new_mapping

    def filter_with_multi_pass(input_report, 
                            categorical_mapping, 
                            category
                            ):
        new_mapping = []
        prompt_tokens, response_tokens = 0, 0
        # for category_stage in categorical_mapping:
        # print(list(category_table[category].keys()))
        # print(category, category_stage)
        final_prompt = assess_categories_multiple(None, input_report, category=category, category_stages=categorical_mapping)
        # final_prompt = assess_category_multiple()
        formatted_chat_history = [
            {"role": "system", "content": "You are a classification assistant that helps classify medical reports. Respond to the following question in JSON format, according to the user's request. Do not include anything else."},
            {"role": "user", "content": final_prompt}
        ]

        result_data = llm_call_model_synchronous(database, llm_ensemble, username, password_prehash, formatted_chat_history, model_choice=model_choice)["model_response"]
        result_data = re.sub(r"^[^\{]*", "", re.sub(r"[^\}]*$", "", result_data))
        try:
            parsed = json.loads(result_data)
            if "report_meets_description" not in parsed:
                assert False, "description validation not in json"
            if parsed["report_meets_description"] != "UNKNOWN":
                parsed["report_meets_description"] = parsed["report_meets_description"].lower()
                parsed["report_meets_description"] = parsed["report_meets_description"][:2].replace(category.lower(), category.upper()) + parsed["report_meets_description"][2:]
            return prompt_tokens, response_tokens, parsed["report_meets_description"]
        except:
            pass
            # print("Failed to parse JSON")
            # new_mapping.append(category_stage)
        return prompt_tokens, response_tokens, None
    # print("Staging function got report text:", report)
    new_results = {}
    for category in list(category_explanations.keys()):
        new_results[category] = {}
        stages_all = list(SAMPLES[category].keys())

        # if not (use_llama == False or use_llama == True): # OpenAI case
        try:
            print("Assessing %17s" % (category), end="\r")
            prompt_tokens_tmp, response_tokens_tmp, result_category = filter_with_multi_pass(report, stages_all, category)
            if result_category is None:
                result_category = "UNKNOWN"
            # prompt_tokens_sample += prompt_tokens_tmp
            # response_tokens_sample += response_tokens_tmp
        except:
            print("Error occured")
            result_category = "UNKNOWN"
        
        if result_category not in stages_all:
            # result_category = "UNKNOWN - " + result_category # Re-enable this later.
            result_category = "UNKNOWN"
        
        new_results[category] = result_category
        # predictions[category].append(result_category)

    ref_results = {}
    for key, value in new_results.items():
        if value == "UNKNOWN":
            ref_results[key] = key+"0"
            continue
        match_get = list(re.finditer(r"(?i)((T|N|M)(\d|o|O|l|L|i|I|x|X))", value[:min(3, len(value))]))[0].group(0).upper()
        match_get = re.sub(r"(?i)(O)", "0", re.sub(r"(?i)(I)", "1", re.sub(r"(?i)(L)", "1", match_get)))
        match_get = match_get[:1] + match_get[1:].lower()
        ref_results[key] = match_get
        if key == "N" and value == "cN1mi":
            ref_results[key] = "N1mi"
        elif match_get[1] == "x": # Replace x category with 0 category for the purpose of determining stage.
            match_get = match_get[:1] + "0"
            ref_results[key] = match_get
    
    ref_results = [ref_results["T"], ref_results["N"], ref_results["M"]]

    if ref_results == ["Tx", "N0", "M0"]:
        stage = "Stage 0"
    elif ref_results == ["T1", "N0", "M0"]:
        stage = "Stage 1A"
    elif ref_results == ["T0", "N1mi", "M0"] or ref_results == ["T1", "N1mi", "M0"]:
        stage = "Stage 1B"
    elif ref_results == ["T0", "N1", "M0"] or \
        ref_results == ["T1", "N1", "M0"] or \
        ref_results == ["T2", "N0", "M0"]:
        stage = "Stage 2A"
    elif ref_results == ["T2", "N1", "M0"] or \
        ref_results == ["T3", "N0", "M0"]:
        stage = "Stage 2B"
    elif ref_results == ["T0", "N2", "M0"] or \
        ref_results == ["T1", "N2", "M0"] or \
        ref_results == ["T2", "N2", "M0"] or \
        ref_results == ["T3", "N1", "M0"] or \
        ref_results == ["T3", "N2", "M0"]:
        stage = "Stage 3A"
    elif ref_results == ["T4", "N0", "M0"] or \
        ref_results == ["T4", "N1", "M0"] or \
        ref_results == ["T4", "N2", "M0"]:
        stage = "Stage 3B"
    elif ref_results[1] == "N3" and ref_results[2] == "M0":
        stage = "Stage 3C"
    elif ref_results[2] == "M1":
        stage = "Stage 4"
    else:
        stage = "Logic Unclear"

    return {
        "title": report_title,
        "t": new_results["T"],
        "n": new_results["N"],
        "m": new_results["M"],
        "stage": stage
    }

def convert_dict_list_to_markdown(list_in : list):
    """
    Convert a dictionary list to a markdown string.
    List must have a first entry describing the order of keys.
    """
    key_ordering = list_in[0]
    entries = list_in[1:]
    keys_create = [{"key": key, "location": value} for key, value in key_ordering.items()]
    keys_create_sorted = sorted(keys_create, key=lambda x: x["location"])
    table_row_strings = ["| "+" | ".join([key_entry["key"] for key_entry in keys_create_sorted]).strip()+" |"]
    table_row_strings.append("| "+" | ".join(["---" for _ in range(len(keys_create_sorted))]).strip()+" |")
    for value_entry in entries:
        new_row_values = []
        for key_entry in keys_create_sorted:
            new_row_values.append(value_entry[key_entry["key"]])
        table_row_strings.append("| "+" | ".join(new_row_values).strip()+" |")
    return {
        "markdown_string": "\n".join(table_row_strings)
    }

def download_dict_list_as_csv(list_in : list, return_string : bool = False):
    key_ordering = list_in[0]
    entries = list_in[1:]
    keys_create = [{"key": key, "location": value} for key, value in key_ordering.items()]
    keys_create_sorted = sorted(keys_create, key=lambda x: x["location"])
    table_rows = [[key_entry["key"] for key_entry in keys_create_sorted]]
    for value_entry in entries:
        new_row_values = []
        for key_entry in keys_create_sorted:
            new_row_values.append(value_entry[key_entry["key"]])
        table_rows.append(new_row_values)
    bytes_target = StringIO()
    # for row in table_rows:
        
    # l = ['list,','of','["""crazy"quotes"and\'',123,'other things']

    line = StringIO()
    writer = csv.writer(line)
    for row in table_rows:
        writer.writerow(row)
    csvcontent = line.getvalue()
    print(csvcontent)
    if return_string:
        return csvcontent
    else:
        bytes_target = BytesIO(bytes(csvcontent, encoding="utf-8"))
        # bytes_target.write(bytes(csvcontent, encoding="utf-8"))
        return {"file_bytes": bytes_target, "file_name": "pTNM_Staging.csv"}
        

    # writer = csv.writer(bytes_target)
    
    # writer.writerows(table_rows)
    # print("Done writing")
    # print(BytesIO.read())
    
if __name__ == "__main__":
    download_dict_list_as_csv([
        {
            "Report Title": 0,
            "T Value": 1,
            "N Value": 2,
            "M Value": 3
        },
        {
            "Report Title": "TNBC0262_Redacted.pdf",
            "T Value": "Tx",
            "N Value": "cNx",
            "M Value": "Mx"
        }
    ])



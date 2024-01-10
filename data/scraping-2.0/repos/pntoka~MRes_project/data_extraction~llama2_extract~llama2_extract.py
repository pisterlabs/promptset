from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json
import os

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="/home/ptoka/llama-2-13b-chat.ggmlv3.q2_K.bin",
    input={"temperature": 0.4, "max_length": 2000, "top_p": 1},
    n_ctx = 2048,
    n_threads = 10
    # callback_manager=callback_manager
    # verbose=True,
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    '''
    Function to structure prompts
    '''
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def temp_time_output_parser(output):
    '''
    Function to parse temp extraction output from Llama-2
    '''
    try:
        temps_str = output.split(':')[1]
    except:
        return None
    temp_values = list(map(str.strip, temps_str.split(',')))
    return temp_values

def temp_time_parser(output, parameter):
    values = []
    units = []
    for val in output:
        try:
            values.append(int(val.split(' ')[0]))
        except:
            continue
    for val in output:
        try:
            units.append(val.split(' ')[1])
        except:
            continue
    units = list(set(units))
    param_dict = {f'{parameter}_values':{'values': values, 'units' : units}}
    return param_dict

def llama_temp_extract(para):
    '''
    Function to extract temperature from a paragraph using Llama-2
    '''
    system_prompt = """You are an expert in materials chemistry called Extractor and are extracting data from fragments of scientific texts. You are interested in extracting the temperatures at which the autoclave was heated in the process described in the text.
    A User will pass in a fragment of text and you will have to return the temperature values and units of the process in the autoclave as a comma separated list. Do not include temperature values of processes not carried out in an autoclave.
    ONLY return the temperature and units in a comma separated list and nothing more. If there is only one relevant temperature value than return just that one. Do not add any explanation of the units. Do not reply with additional conversational text such as "Sure! I can do that".
    
    Here are some examples of previous interactions between you and the User:

    User: Extract the autoclave temperatures as a comma separated list from the following text: The hydrothermal reactor was kept at 200 °C for 8 h and then cooled to room temperature at atmospheric pressure.
    Extractor: 200 °C
    ```
    User: Extract the autoclave temperatures as a comma separated list from the following text: dissolved in 16 mL of distilled water, stirred for 10 min, and then the mixture was transferred to a 30 mL Teflon autoclave, and heated at 260, 280 or 300 °C for 6, 7 or 8 h.
    Extractor: 160 °C, 180 °C, 200 °C
    ```
    User: Extract the autoclave temperatures as a comma separated list from the following text: was added to 30 mL of deionized water and heated at 70 °C for 30 min. Subsequently, the solution was transferred to a Teflon-lined stainless steel autoclave reactor and heated at 180 °C for 12 h
    Extractor: 180 °C
    """
    instruction = "Extract the autoclave temperatures as a comma separated list from the following text: {input}"
    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(template = template, input_variables=["input"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(input=para)
    temp_list = temp_time_output_parser(result)
    if temp_list == None:
        temp_dict = {'temp_values': {'values': [], 'units': []}}
        return temp_dict, result
    temp_dict = temp_time_parser(temp_list, 'temp')
    return temp_dict, result

def llama_time_extract(para):
    '''
    Function to extract time from a paragraph using Llama-2
    '''
    system_prompt= """You are an expert in materials chemistry called Extractor and are extracting data from fragments of scientific texts. You are interested in extracting the time for which the autoclave was kept at a high temperature in the process described in the text.
    A User will pass in a fragment of text and you will have to return the time values and units of the process in the autoclave as a comma separated list. Do not include time values of any processes that are not carried out in an autoclave or ramp times. Do not include any information about temperature.
    ONLY return the time and units in a comma separated list and nothing more. If there is only one relevant time value than return just that one. Do not add any explanation of the units. Do not reply with additional conversational text such as "Sure! I can do that".
    
    Here are some examples of previous interactions between you and the User:
    
    User: Extract the time of the process in the autoclave as a comma separated list from the following text: The hydrothermal reactor was kept at 200 °C for 8 h and then cooled to room temperature at atmospheric pressure.
    Extractor: 8 h
    ```
    User: Extract the time of the process in the autoclave as a comma separated list from the following text: dissolved in 16 mL of distilled water, stirred for 10 min, and then the mixture was transferred to a 30 mL Teflon autoclave, and heated at 260, 280 or 300 °C for 5, 9 or 13 h.
    Extractor: 5 h, 9 h, 13 h
    ```
    User: Extract the time of the process in the autoclave as a comma separated list from the following text: was added to 30 mL of deionized water and heated at 70 °C for 1 h. Subsequently, the solution was transferred to a Teflon-lined stainless steel autoclave reactor and heated at 180 °C for 12 h
    Extractor: 12 h
    """
    instruction = "Extract the time of the process in the autoclave as a comma separated list from the following text: {input}"
    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(template = template, input_variables=["input"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(input=para)
    time_list = temp_time_output_parser(result)
    if time_list == None:
        time_dict = {'time_values': {'values': [], 'units': []}}
        return time_dict, result
    time_dict = temp_time_parser(time_list, 'time')
    return time_dict, result

def file_reader(file_dir):
    '''
    Function to read a a file containg paragraphs and return doi and paragraph text
    '''
    with open(file_dir, 'r') as f:
        paras = []
        dois = []
        for line in f:
            paras.append(line.split(':',1)[1].strip())
            dois.append(line.split(':',1)[0])
    return dois, paras


def llama_temp_time_extract(file_dir, save_dir):
    '''
    Function to extract temperature and time from a file using Llama-2
    '''
    dois, paras = file_reader(file_dir)
    synth_dict_all = []
    for i in range(len(paras)):
        synth_dict = {}
        temp_dict, temp_result = llama_temp_extract(paras[i])
        time_dict, time_result = llama_time_extract(paras[i])
        synth_dict['DOI'] = dois[i]
        synth_dict.update(temp_dict)
        synth_dict.update(time_dict)
        synth_dict['temp_result'] = temp_result
        synth_dict['time_result'] = time_result
        synth_dict_all.append(synth_dict)
    with open(os.path.join(save_dir), 'w') as f:
        json.dump(synth_dict_all, f, indent=4, sort_keys=True, ensure_ascii=False)
    return synth_dict_all
    



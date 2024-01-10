from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import tqdm
import os
import json

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
    
class SemicolSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a semicolon separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split("; ")

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

def material_parser(output, category):
    materials = list(set(output))
    material_dict = {f'{category}_materials': materials}
    return material_dict

def gpt_extract_temp(para):
    '''
    Function to extract reaction temperature as a list using ChatGPT API
    '''
    examples_temp = [
    {"input": "The hydrothermal reactor was kept at 200 °C for 8 h and then cooled to room temperature at atmospheric pressure.", "output": "200 °C"},
    {"input": "The sealed autoclaves were heated to 180 °C in 20 min and kept for 8 h.", "output": "180 °C"},
    {"input":"dissolved in 16 mL of distilled water, stirred for 10 min, and then the mixture was transferred to a 30 mL Teflon autoclave, and heated at 160, 180 or 200 °C for 6, 7 or 8 h.", "output": "160 °C, 180 °C, 200 °C"},
    {"input":"was added to 30 mL of deionized water and heated at 70 °C for 30 min. Subsequently, the solution was transferred to a Teflon-lined stainless steel autoclave reactor and heated at 180 °C for 12 h", "output":"180 °C"}
    ]
    template = """You are an expert in materials chemistry and are extracting data from fragments of scientific texts. You are interested in extracting the temperatures at which the autoclave was heated in the process described in the text.
    A user will pass in a fragment of text and you will have to return the temperature values and units of the process in the autoclave as a comma separated list. Do not include temperature values of processes not carried out in an autoclave.
    ONLY return the temperature and units in a comma separated list and nothing more.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(template)
    human_prompt = "{input}"
    example_prompt = HumanMessagePromptTemplate.from_template("{input}") + AIMessagePromptTemplate.from_template("{output}")
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples = examples_temp)
    final_prompt = (system_prompt + few_shot_prompt + human_prompt)
    chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = final_prompt,
    output_parser = CommaSeparatedListOutputParser()
    )
    result = chain.run({'input':para})
    temp_dict = temp_time_parser(result, 'temp')
    return temp_dict

def gpt_extract_time(para):
    '''
    Function to extract reaction time as a list using ChatGPT API
    '''
    examples_time = [
    {"input": "The hydrothermal reactor was kept at 200 °C for 8 h and then cooled to room temperature at atmospheric pressure.", "output": "8 h"},
    {"input": "The sealed autoclaves were heated to 180 °C in 20 min and kept for 8 h.", "output": "8 h"},
    {"input":"dissolved in 16 mL of distilled water, stirred for 10 min, and then the mixture was transferred to a 30 mL Teflon autoclave, and heated at 160, 180 or 200 °C for 6, 7 or 8 h.", "output": "6 h, 7 h, 8 h"},
    {"input":"was added to 30 mL of deionized water and heated at 70 °C for 30 min. Subsequently, the solution was transferred to a Teflon-lined stainless steel autoclave reactor and heated at 180 °C for 12 h", "output":"12 h"}
    ]
    template = """You are an expert in materials chemistry and are extracting data from fragments of scientific texts. You are interested in extracting the time for which the autoclave was kept at high temperarure in the process described in the text.
    A user will pass in a fragment of text and you will have to return the time values and units of the process in the autoclave as a comma separated list. Do not include time values of processes not carried out in an autoclave or the ramp time.
    ONLY return the temperature and units in a comma separated list and nothing more.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(template)
    human_prompt = "{input}"
    example_prompt = HumanMessagePromptTemplate.from_template("{input}") + AIMessagePromptTemplate.from_template("{output}")
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples = examples_time)
    final_prompt = (system_prompt + few_shot_prompt + human_prompt)
    chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = final_prompt,
    output_parser = CommaSeparatedListOutputParser()
    )
    result = chain.run({'input':para})
    temp_dict = temp_time_parser(result, 'time')
    return temp_dict

def gpt_extract_precursors(para):
    '''
    Function to extract precursor names as a list using ChatGPT API
    '''
    examples_precursors = [
        {"input": "typical pyrolysis process, 0.35 g methionine is added into 30 ml ultra-pure water. After stirring drastically for several minutes, the mixture becomes a homogeneous solution and is transferred into a Teflon-lined autoclave.", "output": "methionine"},
        {"input": "Fluorescent SNCDs were prepared by the solvothermal treatment of p-phenylenediamine and cysteamine hydrochloride", "output": "p-phenylenediamine; cysteamine hydrochloride"},
        {"input": "The N,S-CDs were synthesized by adding 0.0639 g of thiourea, 0.1513 g of urea, and 0.1646 g of sodium citrate into 30 mL of distilled water.", "output":"thiourea; urea; sodium citrate"}
    ]
    template = """You are an expert materials chemisty and are extracting information from fragments of scientific texts which are paragraphs describing a synthetic process of a material or compound. You are interested in extracting the chemical names of the precursors or substrates used in the synthesis process.
    A user will pass in a fragment of text and you will have to return the chemical names of the precursors or substrates as a semicolon separated list. Do not include the names of the solvents used or the form of the compound such as powder or solution. Do not include the names of the target products.
    ONLY return the precursor chemical names in a semicolon separated list and nothing more.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(template)
    human_prompt = "{input}"
    example_prompt = HumanMessagePromptTemplate.from_template("{input}") + AIMessagePromptTemplate.from_template("{output}")
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples = examples_precursors)
    final_prompt = (system_prompt + few_shot_prompt + human_prompt)
    chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = final_prompt,
    output_parser = SemicolSeparatedListOutputParser()
    )
    result = chain.run({'input':para})
    precursor_dict = material_parser(result, 'precursor')
    return precursor_dict

def gpt_extract_targets(para):
    '''
    Function to extract target material names as a list using ChatGPT API
    '''
    examples_targets = [
        {"input":"The N,S-CDs were synthesized by adding 0.0639 g of thiourea, 0.1513 g of urea, and 0.1646 g of sodium citrate into 30 mL of distilled water.", "output": "N,S-CDs"},
        {"input":"CDs were synthesized using a one-step hydrothermal method.", "output": "CDs"}
    ]
    template = """You are an expert in materials chemistry and are extracting information from fragments of scientific texts which are paragraphs describing a synthetic process of a target material or product. You are interested in extracting the chemical names of the target products of the synthesis process being described in the text.
    A user will pass in a fragment of text and you will have to return the chemical names of the targets or products as a semicolon separated list. Do not include the name of precursors used or the form of the compound such as powder or solution. Do not include the names of the solvents used.
    ONLY return the target product chemical names in a semicolon separated list and nothing more.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(template)
    human_prompt = "{input}"
    example_prompt = HumanMessagePromptTemplate.from_template("{input}") + AIMessagePromptTemplate.from_template("{output}")
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples = examples_targets)
    final_prompt = (system_prompt + few_shot_prompt + human_prompt)
    chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = final_prompt,
    output_parser = SemicolSeparatedListOutputParser()
    )
    result = chain.run({'input':para})
    precursor_dict = material_parser(result, 'target')
    return precursor_dict

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

def gpt_extract_synth(file_dir, save_dir):
    '''
    Function to extract all synthesis information from a file containing paragraphs using ChatGPT API
    '''
    dois, paras = file_reader(file_dir)
    synth_dict_all = []
    for i in tqdm.tqdm(range(len(paras)), desc = 'Extracting synthesis information'):
        synth_dict = {}
        para = paras[i]
        doi = dois[i]
        temp_dict = gpt_extract_temp(para)
        time_dict = gpt_extract_time(para)
        precursor_dict = gpt_extract_precursors(para)
        target_dict = gpt_extract_targets(para)
        synth_dict['DOI'] = doi
        synth_dict.update(temp_dict)
        synth_dict.update(time_dict)
        synth_dict.update(precursor_dict)
        synth_dict.update(target_dict)
        synth_dict_all.append(synth_dict)
    with open(os.path.join(save_dir), 'w') as f:
        json.dump(synth_dict_all, f, indent=4, sort_keys=True, ensure_ascii=False)
    return synth_dict_all

def QY_output_parser(output):
    if output == ['No value']:
        QY_dict = {'QY':{'values':[], 'units':[]}}
        return QY_dict
    values = []
    units = []
    for val in output:
        try:
            values.append(float(val.split('%')[0]))
            units.append('%')
        except:
            continue
    units = list(set(units))
    QY_dict = {'QY':{'values':values, 'units':units}}
    return QY_dict

def gpt_extract_QY(para):
    ''''
    Function to extract QY values from a paragraph using ChatGPT API
    '''
    examples_QY = [
        {"input": "could be passivated to achieve high quantum yield (44%)", "output": "44%"},
        {"input": "the PLQY value decreases from 97.4% to 80.6%", "output": "97.4%, 80.6%"},
        {"input": "The fluorescent quantum yields were thus calculated to be 19.3% and 24.1% using quinine sulfate as a reference.", "output": "19.3%, 24.1%"},
        {"input": "The QY of the as-prepared CDs was 0.34", "output": "0.34"},
        {"input": "The value of the QY increased significantly.", "output": "No value"}
    ]
    template = """You are an expert in chemistry and are extracting information from fragments of scientific text which are paragraphs discussing results. You are interested in extracting the values of the quantum yield (QY), also known as PLQY, reported in the text.
    A user will pass in a fragment of text and you will have to return the values of the quantum yield reported in the text as a comma separated list. Do not include the quantum yield value of the reference or standard material.
    ONLY return the quantum yield values in a comma separated list and nothing more. If there is no value of the quantum yield in the fragment of text respond by saying 'No value'.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(template)
    human_prompt = "{input}"
    example_prompt = HumanMessagePromptTemplate.from_template("{input}") + AIMessagePromptTemplate.from_template("{output}")
    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples = examples_QY)
    final_prompt = (system_prompt + few_shot_prompt + human_prompt)
    chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = final_prompt,
    output_parser = CommaSeparatedListOutputParser()
    )
    result = chain.run({'input':para})
    QY_dict = QY_output_parser(result)
    return QY_dict

def gpt_extract_all_QY(file_dir, save_dir):
    '''
    Function to extract QY values from a file containing paragraphs using ChatGPT API
    '''
    dois, paras = file_reader(file_dir)
    QY_dict_all = []
    for i in tqdm.tqdm(range(len(paras)), desc = 'Extracting QY information'):
        QY_dict = {}
        para = paras[i]
        doi = dois[i]
        QY_dict = gpt_extract_QY(para)
        QY_dict['DOI'] = doi
        QY_dict_all.append(QY_dict)
    with open(os.path.join(save_dir), 'w') as f:
        json.dump(QY_dict_all, f, indent=4, sort_keys=True, ensure_ascii=False)
    return QY_dict_all









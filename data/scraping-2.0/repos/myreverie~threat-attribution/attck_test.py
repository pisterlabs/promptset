from typing import List
from stix2 import FileSystemSource, Filter
from mitreattack.stix20 import MitreAttackData
from pprint import pprint
import json
import re
import requests
from time import sleep
import Levenshtein
import networkx as nx
import community
import matplotlib.pyplot as plt
from typing import List
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')
API_KEY = config.get('GPT', 'api_key')


examples = [
    {
        'example_input': 'The adversary renamed ntdsaudit.exe to msadcs.exe.',
        'example_output': '{"Slot": [{"Renamed Utilities": ["ntdsaudit.exe"]}, {"Renamed Strings": ["msadcs.exe"]}]}'
    },
    {
        'example_input': 'APT28 has used a variety of public exploits, including CVE 2020-0688 and CVE 2020-17144, to gain execution on vulnerable Microsoft Exchange; they have also conducted SQL injection attacks against external websites.',
        'example_output': '{"Slot": [{"CVE-ID": ["CVE 2020-0688"]}, {"Exploited Vulnerablility Type": ["SQL injection attacks"]}, {"Vulnerable Programs": ["Microsoft Exchange"]}]}'
    },
    {
        'example_input': 'An APT3 downloader creates persistence by creating the following scheduled task: schtasks /create /tn "mysc" /tr C:\\Users\Public\\test.exe /sc ONLOGON /ru "System".',
        'example_output':'{"Slot": [{"Task Name": ["mysc"]}, {"Task Run": ["C:\\Users\\Public\\test.exe"]}, {"Schedule Type": ["User"]}]}'
    },
    {
        'example_input': 'STARWHALE has the ability to create the following Windows service to establish persistence on an infected host: sc create Windowscarpstss binpath= "cmd.exe /c cscript.exe c:\\windows\\system32\\w7_1.wsf humpback_whale" start= "auto" obj= "LocalSystem".',
        'example_output': '{"Slot": [{"Service Name": ["Windowscarpstss"]}, {"Binary Path": ["cmd.exe /c cscript.exe c:\\windows\\system32\\w7_1.wsf humpback_whale"]}, {"Start Type": ["auto"]}, {"Service Account": ["LocalSystem"]}]}'
    },
    {
        'example_input': 'Dragonfly has used VPNs and Outlook Web Access (OWA) to maintain access to victim networks.',
        'example_output': '{"Slot": [{"Access Method": ["VPNs", "Outlook Web Access (OWA)"]}]}'
    }
]
class CandicatedTTPSchemaGeneratePromptTemplate(object):
    def __init__(self, examples) -> None:
        introduction_template = """Your goal is to generalizing a schema of attack techniques from the given input of attack technique descriptions.

All output must be in JSON format and follow the pattern specified above. Do not output anything except for the extracted information. Do not add any clarifying information. Do not add any fields that are not in the pattern. If the text contains attributes that do not appear in the pattern, please ignore them.
Here is the output pattern:
```
{{"properties": {{"Slot": {{"items": {{"type": "object"}}, "title": "Slot", "type": "array"}}}}, "required": ["Slot"]}}
```"""
        introduction_prompt = PromptTemplate.from_template(introduction_template)
        # 需要对examples和template里的花括号进行转义，否则会被当成模板变量报KeyError
        for example in examples:
            example['example_output'] = example['example_output'].replace('{', '{{').replace('}', '}}')
        example_template = """Input: {example_input}
Output: {example_output}"""
        example_prompt = PromptTemplate(input_variables=["example_input", "example_output"], template=example_template)
        demonstrations_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix='Here are some positive examples:',
            example_separator='\n\n',
            input_variables=[],
            suffix=''
        )

        start_template = """Now, do this for real. Here is the input, please generate just one output without any additional information:
Input: {input}
Output: """
        start_prompt = PromptTemplate.from_template(start_template)

        full_template = """{introduction}

{demonstrations}

{start}"""
        full_prompt = PromptTemplate.from_template(full_template)
        input_prompts = [
            ("introduction", introduction_prompt),
            ("demonstrations", demonstrations_prompt),
            ("start", start_prompt)
        ]
        self.candicated_TTP_schema_generate_pipeline_prompts = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    def get_prompt_string(self, input_text) -> str:
        return self.candicated_TTP_schema_generate_pipeline_prompts.format(input=input_text)

class CandicatedTTPScheme(BaseModel):
    Slot: List[dict]

json_parser = PydanticOutputParser(pydantic_object=CandicatedTTPScheme)

def is_valid_format(text, parser):
    try:
        parser.parse(text)
        json.loads(text)
        return True
    except ValueError:
        return False
    except json.decoder.JSONDecodeError:
        return False
def get_result_from_gpt(text, api_key, temperature=0.4, n=5):
    prompt_template = CandicatedTTPSchemaGeneratePromptTemplate(examples)
    prompt_string = prompt_template.get_prompt_string(text)
    candicated_template_list = []

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    json_data = {
        'model': 'gpt-3.5-turbo',
        'temperature': temperature,
        'messages': [
            {
                'role': 'user',
                'content': prompt_string,
            },
        ],
    }
    for i in range(n):
        response = requests.post('https://api.ai.cs.ac.cn/v1/chat/completions', headers=headers, json=json_data)
        print(response.text)
        if response.text == '':
            print('GPT接口返回空字符串')
            continue
        result = json.loads(response.text)
        if 'error' in result.keys():
            print('GPT接口返回错误')
            continue
        result = json.loads(response.text)['choices'][0]['message']['content']
        result = result.replace('\\', '\\\\')
        if is_valid_format(result, json_parser):
            print('输出格式正确')
            candicated_template = json.loads(result)
            candicated_template_list.append(candicated_template)
            sleep(1)
        else:
            print('输出格式有误')

    return candicated_template_list


def get_procedure_example(src, techniques_object_id) -> List:
    """根据技术的stix id，找到所有的程序实例"""
    procedure_example_list = []
    query_results = src.query([
        Filter('target_ref', '=', techniques_object_id),
        Filter('relationship_type', '=', 'uses'),
        Filter('type', '=', 'relationship'),
    ])
    for result in query_results:
        # print('处理前：', result.description)
        processed_description = procedure_text_preprocess(result.description)
        # print('处理后：', processed_description)
        procedure_example_list.append(processed_description)
    return procedure_example_list


def procedure_text_preprocess(text):
    """处理ATT&CK的markdown文本"""
    link_pattern = r'\[(.*?)\]\(.*?\)'
    link_replacement = r'\1'
    code_pattern = r'<code>(.*?)</code>'
    code_replacement = r'\1'
    citation_pattern = r'\(Citation: .*?\)'
    citation_replacement = r''
    backquote_pattern = r'`(.*?)`'
    backquote_replacement = r'\1'

    text = re.sub(link_pattern, link_replacement, text)
    text = re.sub(code_pattern, code_replacement, text)
    text = re.sub(citation_pattern, citation_replacement, text)
    text = re.sub(backquote_pattern, backquote_replacement, text)

    return text


def test1():
    mitre_attack_data = MitreAttackData("./data/cti/enterprise-attack/enterprise-attack.json")
    src = FileSystemSource('./cti/enterprise-attack')
    techniques = mitre_attack_data.get_techniques()
    pprint(techniques[0])
    for technique in techniques:
        print(technique.id, technique.external_references[0].external_id)
        get_procedure_example(src, technique.id)


def test_by_specific_technique(technique_id):
    mitre_attack_data = MitreAttackData("./data/cti/enterprise-attack/enterprise-attack.json")
    src = FileSystemSource('./cti/enterprise-attack')
    technique_object_id = mitre_attack_data.get_object_by_attack_id(technique_id, 'attack-pattern').id
    print(technique_object_id)
    procedure_example_list = get_procedure_example(src, technique_object_id)
    template_list = []
    for index, procedure_example in enumerate(procedure_example_list):
        print(f'第{index}/{len(procedure_example_list)}个程序示例')
        print(procedure_example)
        candicated_template_list = get_result_from_gpt(procedure_example, API_KEY)
        template_list.append(candicated_template_list)
        print(candicated_template_list)
    print('-----------------')
    print(json.dumps(template_list))


def test2():
    get_result_from_gpt('HAFNIUM has checked for network connectivity from a compromised host using ping, including attempts to contact google[.]com.', API_KEY)


def template_aggregate(template_json):
    template_list_from_diff_text = json.loads(template_json)
    aggregated_template_list_from_diff_text = []
    for template_list_from_same_text in template_list_from_diff_text:
        candicated_template_slot_list = []
        for template in template_list_from_same_text:
            candicated_template_slot_list.extend(template['Slot'])
        graph = build_similarity_graph(candicated_template_slot_list)
        clusters = louvain_clustering(graph, resolution=1.45)
        aggregated_template_list = []
        for cluster in clusters:
            template_keys_set = set()
            template_values_set = set()
            for node_id in cluster:
                node_data = json.loads(graph.nodes(data=True)[node_id]['label'])
                template_keys_set.update(node_data.keys())
                template_values_set.update(list(node_data.values())[0])
            aggregated_template_list.append((template_keys_set, template_values_set))
        aggregated_template_list_from_diff_text.append(aggregated_template_list)
    for i in aggregated_template_list_from_diff_text:
        for j in i:
            print(j)
        print('---')


def calc_similarity_between_str_list(str_list1: List[str], str_list2: List[str]) -> float:
    """计算两个字符串列表的相似度，两两比较取最大值"""
    edit_distance_similarity = 0.0
    for str1 in str_list1:
        for str2 in str_list2:
            edit_distance = Levenshtein.distance(str1, str2)
            temp_result = 1 - edit_distance/max(len(str1), len(str2))
            if temp_result > edit_distance_similarity:
                edit_distance_similarity = temp_result
    return edit_distance_similarity


def build_similarity_graph(candicated_template_slot_list):
    graph = nx.Graph()
    for i in range(len(candicated_template_slot_list)):
        graph.add_node(i, label=json.dumps(candicated_template_slot_list[i]))
    for i in range(len(candicated_template_slot_list)):
        for j in range(i + 1, len(candicated_template_slot_list)):
            distance = calc_similarity_between_str_list(list(candicated_template_slot_list[i].values())[0], list(candicated_template_slot_list[j].values())[0])
            print(candicated_template_slot_list[i], candicated_template_slot_list[j], distance)
            graph.add_edge(i, j, weight=distance)
    return graph


def louvain_clustering(graph, resolution=1):
    partition = community.best_partition(graph, resolution=resolution)
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    return list(clusters.values())


if __name__ == '__main__':
    # test_by_specific_technique('T1053.005')
    # template_aggregate('[[{"Slot":[{"Signed Status":["incompletely signed"]},{"Certificate Status":["revoked"]}]},{"Slot":[{"Signed Status":["incompletely signed"]},{"Certificate Status":["revoked"]},{"Malware Name":["WindTail"]}]},{"Slot":[{"Signed Status":["incompletely signed"]},{"Certificate Status":["revoked"]}]}],[{"Slot":[{"Signed with Authenticode Certificate":["Invalid"]}]},{"Slot":[{"Signed with":["invalid Authenticode certificate"]}]},{"Slot":[{"Malware Name":["BADNEWS"]},{"Signature":["invalid Authenticode certificate"]},{"Purpose":["look more legitimate"]}]}],[{"Slot":[{"Stage":["1"]},{"Module Type":["64-bit"]},{"Signed with Fake Certificates":["Microsoft Corporation","Broadcom Corporation"]}]},{"Slot":[{"Stage":["1"]},{"Module Type":["64-bit"]},{"Signed Certificates":["fake certificates masquerading as originating from Microsoft Corporation and Broadcom Corporation"]}]},{"Slot":[{"Stage":["1"]},{"Module Type":["64-bit"]},{"Signed Certificates":["fake certificates"]},{"Certificate Origin":["Microsoft Corporation","Broadcom Corporation"]}]}],[{"Slot":[{"Malware Signature":["invalid digital certificates"]},{"Certificate Issuer":["Tencent Technology (Shenzhen) Company Limited"]}]},{"Slot":[{"Malware Signature":["Tencent Technology (Shenzhen) Company Limited."]}]},{"Slot":[{"Malware Signature":["Tencent Technology (Shenzhen) Company Limited."]}]}],[{"Slot":[{"Tactic":["Code Signing"]}]},{"Slot":[{"Tactic":["Code Signing"]},{"Technique":["Revoked Certificates"]},{"Group":["Windshift"]}]},{"Slot":[{"Tactic":["Code Signing"]},{"Technique":["Use of revoked certificates"]},{"Group Name":["Windshift"]}]}],[{"Slot":[{"Signed By":["fake and invalid digital certificates"]}]},{"Slot":[{"Signed By":["fake and invalid digital certificates"]}]},{"Slot":[{"Signed by":["fake and invalid digital certificates"]}]}],[{"Slot":[{"Malicious Activity":["Use of invalid certificate"]}]},{"Slot":[{"Certificate":["invalid"]}]},{"Slot":[{"Malicious Activity":["Use of invalid certificate"]}]}],[{"Slot":[{"Malware Name":["Gelsemium"]},{"Tactic Used":["Unverified signatures on malicious DLLs"]}]},{"Slot":[{"Malware Name":["Gelsemium"]},{"Tactic":["Unverified signatures on malicious DLLs"]}]},{"Slot":[{"Malware Name":["Gelsemium"]},{"Tactic":["Unverified Signatures"]}]}]]')
    get_result_from_gpt(r'POWERSTATS has established persistence through a scheduled task using the command "C:\Windows\system32\schtasks.exe" /Create /F /SC DAILY /ST 12:00 /TN MicrosoftEdge /TR "c:\Windows\system32\wscript.exe C:\Windows\temp\Windows.vbe".', API_KEY)
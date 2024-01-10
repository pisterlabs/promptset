import sys
from pathlib import Path
from pprint import pprint
_pwd = Path(__file__).absolute()
_pa_path = _pwd.parent.parent
# print(_pa_path)
sys.path.append(str(_pa_path))
# pprint(sys.path)
from module.query_vdb import qa_faiss_multi_query_azure
from dotenv import load_dotenv
load_dotenv()

def writeF(_dir, _fn, _txt):
    import os
    wfn = os.path.join(_dir, _fn)
    with open(wfn, 'w', encoding='utf-8') as wf:
        wf.write(_txt)

def readF(_dir, _fn):
    import os
    rfn = os.path.join(_dir, _fn)
    with open(rfn, 'r', encoding='utf-8') as rf:
        return rf.read()

_override_rules = [
    "A specific disk type can be converted to another disk type if necessary, but such conversion does not change anything about that disk type (features, limitations, comparisons, optimizations, applicability, etc.) other than affecting the feature of changing disk type itself.",
    # "A specific disk type can be converted to another disk type if necessary, but such conversion feature only affect the feature of changing disk type, nothing else.",
    # "Any disk type can be converted to any disk type.",
    # "Any disk can be used for any workload, as long as IOPS and Throughput requirements of source disk has been met."
]

##### 6) check conflict rules
import os, re
from langchain.callbacks import get_openai_callback
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
_dir = 'tmp_1689564655'
# _dir = 'tmp_1689566000'
_original_rules = readF(_dir, '_ans4').split("\n")

def parse_ans7(_ans7):
    # print(_ans7)
    _new_rule, _reason = "", ""
    _l = _ans7.find('(')
    _r = _ans7.find(')')
    _new_rule = _ans7[:_l].strip()
    if _new_rule[-1] != ".":
        _new_rule += "."
    _reason = _ans7[_l+1:_r]
    # print(f"'{_new_rule}'")
    # print(f"'{_reason}'")
    return _new_rule, _reason

# parse_ans7("Any disk can be used for any workload, as long as IOPS and Throughput requirements of source disk has been met. Standard SSDs offer better availability, consistency, reliability, and latency compared to Standard HDDs. (no conflict between the two rules).")

# exit()
merge_template = \
"""
Giving the following overriding rule:
--------------------
{overriding}
--------------------
Giving the following original rule:
--------------------
{original}
--------------------
Please compare the giving overriding rule with the giving original rule, and perform logical analysis to generate the new rule and short reason. If there is some logical conflicts between the two, the new rule must adjust the conflicting part to meet the requirements of the overriding rule, and for the non-conflicting part, keep the requirements in the original rule. If there is NO logical conflict at all between the overriding rule and the original rule, REMEMBER the new rule is only the original rule left unchanged (ignore the orriding rule) and the short reason is 'No conflict'. Based on the above instruction, output the generated new rule and short reason on only one line in the format "ans: new rule. (why: short reason)", nothing else:
"""
merge_prompt = PromptTemplate.from_template(merge_template)
llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
_ans7, _step7, _ans7log = [], [], []
_ans7_str, _step7_str, _ans7log_str = "", "", ""
with get_openai_callback() as cb:
    _over_rule, _orig_rule = "", ""
    for i in _override_rules:
        _over_rule = i
        _new_rules = []
        _ans7.append(f"## {_over_rule}")
        for j in _original_rules:
            _orig_rule = re.sub('^\d+\. ', '', j)
            chain = LLMChain(llm=llm, prompt=merge_prompt)
            _re = chain.run(overriding=_over_rule, original=_orig_rule)
            _re_str = re.sub('^ans: ', '', _re.strip())
            _new_rule, _reason = parse_ans7(_re_str)
            _log = f"orig: '{j}'\nover: '{_over_rule}'\n>>>>: '{_re.strip()}'\n\n"
            print(_log)
            _ans7log.append(_log)
            _step7.append("="*20+" prompt "+"="*20+"\n" + merge_prompt.format(overriding=_over_rule, original=_orig_rule))
            _ans7.append(_new_rule)
            _new_rules.append(_new_rule)
        ##### reset _original_rules
        # _original_rules = _new_rules
    _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    _step7_str = f"{_token_cost}\n\n" + "\n\n".join(_step7)
    _ans7_str = "\n".join(_ans7)
    _ans7log_str = "\n".join(_ans7log)

writeF(_dir, '_ans4_m1', _ans7_str)
writeF(_dir, '_ans4log_m1', _ans7log_str)
writeF(_dir, '_step4_m1', _step7_str)


"""Prompts used for LLM
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# SYSTEM PROMPT
SYSTEM_PROMPT = """
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC).
Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.

ANALYSE the given patient profile based on given query based on one of the following criteria:
- Whether treated patient is new patient or patient under maintenance
- Prior response to Infliximab
- Prior failure to Anti-TNF agents
- Prior failure to Vedolizumab
- Age
- Pregnancy
- Extraintestinale manifestations
- Pouchitis

FINALLY RETURN up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.

=========
REFERENCE TEXT:
Content: Recommendation is to use drug A for age 20-30, drug B or C for age 30-50, use drug D for age above 50.
Source: DS-1
Content: For patient with extraintestinale manifestations, avoid using drug C.
Source: DS-2
Content: For pregnant woman, preferred choice is always drug A.
Source: DS-3
=========
Human: A 34-year-old man with history of extraintestinale manifestation. Let's think step by step.
AI: Let's think step by step. The patient is a 34-year-old man, therefore based on source DS-1, he should use drug B or C (for patient between age 30-50). From source DS-2, as the patient has history of extraintestinale manifestations, he should avoid drug C. Hence the recommended drug is drug B.
[
    {{"drug_name" : "B", "advantages" : "Good for middle age patient", "disadvantages" "NA"}}
]

=========
REFERENCE TEXT:
Content: Recommendation is to use drug A for age 20-30, drug B or C for age 30-50, use drug D for age above 50.
Source: DS-1
Content: For patient with extraintestinale manifestations, avoid using drug C.
Source: DS-2
Content: For pregnant woman, preferred choice is always drug A.
Source: DS-3
=========
Human: A 50-year-old pregnant woman. Let's think step by step.
AI: Let's think step by step. The patient is a 50-year-old woman, therefore based on source DS-1, she should use drug D. However, from source DS-3, pregnant woman should always use drug D. Hence recommended drug is D.
[
    {{"drug_name" : "D", "advantages" : "Good for pregnant woman", "disadvantages" "NA"}}
]

=========
REFERENCE TEXT:
{summaries}
=========
"""

# QUESTION PROMPT
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(
    "{question}\nLet's think step by step."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT, input_variables=["summaries"]
        ),
        HUMAN_PROMPT,
    ]
)

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(PROMPT_TEMPLATE.input_variables)
    print(BaseExperiment.convert_prompt_to_string(PROMPT_TEMPLATE))

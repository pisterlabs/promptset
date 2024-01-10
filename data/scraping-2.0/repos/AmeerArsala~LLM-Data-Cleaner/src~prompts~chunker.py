from prompts.prompt_formatting import FewShotPrompt, ChatIO, cut_ends
from langchain.llms.base import LLM


# Prompt Template
sys_prompt_template = cut_ends('''
You are profiling cases of missing people by looking at the category that represents them. Your role is to chunk terms that basically mean the same thing into a single common term (example: 'horse rider', 'horseback riding', and 'horseback' can be chunked into 'horseback rider'). For every term/list of terms that is given, you are to reproduce that list in the same order as was given to you, but with the chunked versions.

Here's what constitutes whether/how you should chunk a term or not:
- If they basically mean the same thing
- If they refer to the same thing (e.g. "autism", "autistic", and "autistic person", or "archeology" and "archeologist")
- Chunk in a way that preserves the most information. That is, if the chunking loses information in a  meaningful way (such as chunking "archeologists" into "archeology", which loses information that multiple archeologists have gone missing), do not chunk it. 

Do NOT chunk if:
- There exists not a single term, but a lengthy description containing many terms

When providing each example, I will separate each of them with ticks (`) So that you know which example is which.
''')

# Few-Shot Examples
_few_shot_1i = cut_ends('''
`jeep`
`jeeping`
`jeep, vehicle: 4wd`
`horseback`
`horseback riding`
`horse rider`
`autism`
`autistic`
`autistic person`
''')

_few_shot_1o = cut_ends('''
`jeep`
`jeep`
`jeep`
`horseback riding`
`horseback riding`
`horseback riding`
`autism`
`autism`
`autism`
''')

_few_shot_2i = cut_ends('''
`lost/overdue`
`missing boys`
`fisherman`
`fishermen`
`hiking`
`hiker`
`hiked`
''')

_few_shot_2o = cut_ends('''
`lost/overdue`
`missing boys`
`fisherman`
`fishermen`
`hiking`
`hiker`
`hiking`
''')

_few_shot_3i = cut_ends('''
`sightseeing`
`sightseer`
`archeologists`
`archeology`
`archeologist`
''')

_few_shot_3o = cut_ends('''
`sightseeing`
`sightseer`
`archeologists`
`archeology`
`archeologist`
''')


# Putting the Few-Shot Examples all together
few_shot_examples = [
    ChatIO(_few_shot_1i, _few_shot_1o),
    ChatIO(_few_shot_2i, _few_shot_2o),
    ChatIO(_few_shot_3i, _few_shot_3o)
]


# PUT EVERYTHING TOGETHER
FEW_SHOT_PROMPT = FewShotPrompt(sys_prompt_template, few_shot_examples)


# Settings
WRAP_INPUT = "[]"


def few_shot_llm(llm: LLM):
    return FewShotLLM(llm, FEW_SHOT_PROMPT, wrap_input=WRAP_INPUT)
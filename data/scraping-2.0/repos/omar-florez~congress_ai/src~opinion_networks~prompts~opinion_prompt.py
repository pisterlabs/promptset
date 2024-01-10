from langchain.prompts import PromptTemplate

def get_background(background):
    if background is None:
        return "\n"
    background = f"""\nAlign these opinions to the values, experiences, and economic activity related to people with this background: {background}\n"""
    return background.strip()

#------------------------------------------------------------------------------------------------------------------------
read_template = """Considering the following law written in {language}:
{law}

Return two opinions, an opinion in favor of that law and an opinion against that law in this JSON format:

```json
[
{{
"Opinion": "+",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and positive consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}},
{{
"Opinion": "-",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and negative consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}}
]
```

{background}


OUTPUT:
"""

#------------------------------------------------------------------------------------------------------------------------
# reduce_template = """Given this law:
# {law}

# and this list of opinions:
# {opinions}

# Your task is to use your critical thinking to generate a two well-thought and diverse opinions, one in favor and another against that law, combining opinions that are similar, and creating new ones by weighing the evidence presented and associating ideas in such a way as to enhance the original opinions. 
# {background}

# Return the two opinions in the following JSON format:

# ```json
# [
# {{
# "Opinion": "+",
# "Reasoning": "Reasons why you would vote in favor of that law.",
# "Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
# }},
# {{
# "Opinion": "-",
# "Reasoning": "Reasons why you would vote against that law.",
# "Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
# }}
# ]
# ```

# The output should be in {language} and only include the JSON output. 


# OUTPUT:
# """
#------------------------------------------------------------------------------------------------------------------------
summary_template = """Considering these opinions:
{opinions}

Combine them into two opinions, one in favor and another against this law:
{law}

In the following JSON format:

```json
[
{{
"Opinion": "+",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and positive consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}},
{{
"Opinion": "-",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and negative consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}}
]
```

The output should be in {language} and only include the JSON output.


OUTPUT:
"""


reduce_template = """Given this law:
{law}

Reflect on this list of opinions about such law:
{opinions}

Your job is to form your own new opinions by giving more importance to opinions with higher scores, associating the ideas of the original opinions, fact-checking their evidences, analyze their motivations. 
{background}

Return these two opinions, one in favor and another against that law, in the following JSON format:

```json
[
{{
"Opinion": "+",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and positive consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}},
{{
"Opinion": "-",
"Reasoning": "Reasons why you would vote in favor of that law including evidences and negative consequences.",
"Score": "A numeric value between 0.0 and 1.0 indicating how much you are support your opinion."
}}
]
```

The output should be in {language} and only include the JSON output. 


OUTPUT:
"""

#------------------------------------------------------------------------------------------------------------------------
collapse_template = """Given this law:
{law}

and this list of opinions:
{opinions}

Your task is to use your critical thinking to generate a bulleted list of opinions in favor and against the law, combining opinions that are similar, and creating new ones by weighing the evidence presented and associating ideas in such a way as to enhance the original opinions. 

Each item is an opinion with the following format:

Score: A numeric value between 0.0 and 1.0 indicating how much you are support your opinion. Your opinion. 

OUTPUT:
"""

READ_PROMPT = PromptTemplate(input_variables=["law", "background", "language"], template=read_template)
COLLAPSE_PROMPT = PromptTemplate(input_variables=["law", "opinions"], template=collapse_template)
REDUCE_PROMPT = PromptTemplate(input_variables=["law", "opinions", "background", "language"], template=reduce_template)
DECIDE_PROMPT = PromptTemplate(input_variables=["law", "opinions", "background", "language"], template=reduce_template)
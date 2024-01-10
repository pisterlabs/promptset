from langchain import PromptTemplate

FIX_IT_TEMPLATE = """\
Test run failed. Here are the logs.

Stdout:
{stdout}

Stderr:
{stderr}

Fix it.
"""

fix_it_prompt = PromptTemplate.from_template(FIX_IT_TEMPLATE)

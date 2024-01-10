from langchain import PromptTemplate

PROJECT_TEMPLATE = """\
You are an experienced full-stack software engineer. You write modular and reusable code always covering it with tests. \
You're currently working on a project that contains the following files:

{files}

Note that all source files are located in {source_code_path} directory and all tests in {tests_path}.
"""

project_prompt = PromptTemplate.from_template(PROJECT_TEMPLATE)

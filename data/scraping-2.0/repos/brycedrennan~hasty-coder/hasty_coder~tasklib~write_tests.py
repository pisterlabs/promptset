"""
Steps to write a unit test for a single function
  - autoformat code (maintain indentation)
  - obtain context
    - function
      - code
      - list of external references (functions, variables, classes, etc)
    - class
      - signature of parent class and sibling functions if it exists
    - file
        - filepath relative to project root
        - signature of sibling functions in file
        - import statements in the same file
        - docstring of the file
    - project
        - project "readme"
        - signature of any functions used in the function
        - examples of it being called from other parts of the codebase
  - look for gaps in context
    - create missing docstrings
    - create missing readme
  - generate tests
    - summarize code line by line
    - create list of scenarios and subscenarios we want to test
    - create list of rare or unexpected edge cases (with examples)
    - for each test:
       - check if valid code and autoformat
  - output test to proper location


"""

from hasty_coder import openai_cli


def write_test(code_snippet, project_plan=None):
    prompt = f"""
INSTRUCTIONS:
Write unit tests for the following code snippet. Use the pytest library. Use the pytest `monkeypatch` fixture to mock any external calls.

CODE SNIPPET:
```
{code_snippet}
```

UNIT TESTS:"""
    print(prompt)
    test_code = openai_cli.completion(prompt)
    return test_code

#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
import ast

"""
用思维链（CoT），一步一步提示chatgpt，生成更加完整的单元测试代码
先让chatgpt解释代码作用
然后把代码及解释给到chatgpt，让其告知需要多少测试用例
最后让chatgpt生成单元测试代码
"""

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# gpt封装
def gpt35(prompt, model="text-davinci-002", temperature=0.4, max_tokens=1000, 
          top_p=1, stop=["\n\n", "\n\t\n", "\n    \n"]):
    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        stop = stop
        )
    message = response["choices"][0]["text"]
    return message


# 我们需要，用text-davinci-002模型
# pyhton3.10，用pytest包，对输入函数，生成单元测试
# 但首先要先看下代码做了什么事情，
# 并用First给出了引导性提示
def explain_code(function_to_test, unit_test_package="pytest"):
    prompt = f""""# How to write great unit tests with {unit_test_package}

In this advanced tutorial for experts, we'll use Python 3.10 and `{unit_test_package}` to write a suite of unit tests to verify the behavior of the following function.
```python
{function_to_test}


Before writing any unit tests, let's review what each element of the function is doing exactly and what the author's intentions may have been.
- First,"""
    response = gpt35(prompt)
    return response, prompt

"""
- First, we use the `divmod` built-in function to get the quotient and remainder of `seconds` divided by 60. This is assigned to the variables `minutes` and `seconds`, respectively.
- Next, we do the same thing with `minutes` and 60, assigning the results to `hours` and `minutes`.
- Finally, we use string interpolation to return a string formatted according to how many hours/minutes/seconds are left.
"""


# 生成一个测试计划
def generate_a_test_plan(full_code_explaination, unit_test_package="pytest"):
    prompt_to_explain_a_plan = f"""
    
A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of `{unit_test_package}` to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way

`{unit_test_package}` has many convenient features that make it easy to write and maintain unit tests. We'll use them to write unit tests for the function above.

For this particular function, we'll want our unit tests to handle the following diverse scenarios (and under each scenario, we include a few examples as sub-bullets):
-"""
    prompt = full_code_explaination + prompt_to_explain_a_plan
    response = gpt35(prompt)
    return response, prompt

"""
- Normal behavior:
    - `format_time(0)` should return `"0s"`
    - `format_time(59)` should return `"59s"`
    - `format_time(60)` should return `"1min0s"`
    - `format_time(119)` should return `"1min59s"`
    - `format_time(3600)` should return `"1h0min0s"`
    - `format_time(3601)` should return `"1h0min1s"`
    - `format_time(3660)` should return `"1h1min0s"`
    - `format_time(7200)` should return `"2h0min0s"`
- Invalid inputs:
    - `format_time(None)` should raise a `TypeError`
    - `format_time("abc")` should raise a `TypeError`
    - `format_time(-1)` should raise a `ValueError`
"""

"""
- The function is called with a valid number of seconds
    - `format_time(1)` should return `"1s"`
    - `format_time(59)` should return `"59s"`
    - `format_time(60)` should return `"1min"`
- The function is called with an invalid number of seconds
    - `format_time(-1)` should raise a `ValueError`
    - `format_time("60")` should raise a `TypeError`
- The function is called with a `None` value
    - `format_time(None)` should raise a `TypeError`
"""


# 生成测试用例
def generate_test_cases(function_to_test, unit_test_package="pytest"):
    starter_comment = "Below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator"
    prompt_to_generate_the_unit_test = f"""

Before going into the individual tests, let's first look at the complete suite of unit tests as a cohesive whole. We've added helpful comments to explain what each line does.
```python
import {unit_test_package}  # used for our unit tests

{function_to_test}

#{starter_comment}"""
    full_unit_test_prompt = prompt_to_explain_code + code_explaination + test_plan + prompt_to_generate_the_unit_test
    return gpt35(model="text-davinci-003", prompt=full_unit_test_prompt, stop="```"), prompt_to_generate_the_unit_test

"""
import pytest  # used for our unit tests

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"

#Below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator.
#The first element of the tuple is the name of the test case, and the second element is the value to be passed to the format_time() function.
@pytest.mark.parametrize('test_input,expected', [
    ('0', '0s'),
    ('59', '59s'),
    ('60', '1min0s'),
    ('119', '1min59s'),
    ('3600', '1h0min0s'),
    ('3601', '1h0min1s'),
    ('3660', '1h1min0s'),
    ('7200', '2h0min0s'),
])
def test_format_time(test_input, expected):
    #For each test case, we call the format_time() function and compare the returned value to the expected value.
    assert format_time(int(test_input)) == expected

#We use the @pytest.mark.parametrize decorator again to test the invalid inputs.
@pytest.mark.parametrize('test_input', [
    None,
    'abc',
    -1
])
def test_format_time_invalid_inputs(test_input):
    #For each invalid input, we expect a TypeError or ValueError to be raised.
    with pytest.raises((TypeError, ValueError)):
        format_time(test_input)
"""


# 需要生成单元测试的示例代码
code = """
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"
"""


if __name__ == '__main__':
    get_api_key()

    # 让chatgpt解释代码作用
    code_explaination, prompt_to_explain_code = explain_code(code)
    print(code_explaination)

    # 让chatgpt给出测试计划，覆盖哪些用例
    test_plan, prompt_to_get_test_plan = generate_a_test_plan(prompt_to_explain_code + code_explaination)
    print(test_plan)
    
    # 如果用例太少，提醒进行边界测试，生成新的测试用例
    not_enough_test_plan = """The function is called with a valid number of seconds
- `format_time(1)` should return `"1s"`
- `format_time(59)` should return `"59s"`
- `format_time(60)` should return `"1min"`
"""
    approx_min_cases_to_cover = 7
    elaboration_needed = test_plan.count("\n-") +1 < approx_min_cases_to_cover 
    if elaboration_needed:
        prompt_to_elaborate_on_the_plan = f"""

In addition to the scenarios above, we'll also want to make sure we don't forget to test rare or unexpected edge cases (and under each edge case, we include a few examples as sub-bullets):
-"""
    more_test_plan, prompt_to_get_test_plan = generate_a_test_plan(prompt_to_explain_code + code_explaination + not_enough_test_plan + prompt_to_elaborate_on_the_plan)
    print(more_test_plan)
    
    # 生成测试用例
    unit_test_response, prompt_to_generate_the_unit_test = generate_test_cases(code)
    print(unit_test_response)

    # 检查语法
    code_start_index = prompt_to_generate_the_unit_test.find("```python\n") + len("```python\n")
    code_output = prompt_to_generate_the_unit_test[code_start_index:] + unit_test_response
    try:
        ast.parse(code_output)
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
    
    print(code_output)

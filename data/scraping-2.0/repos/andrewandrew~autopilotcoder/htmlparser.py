from llms import openai_wrapper
from prompts import codeprompts
from helpers import execute_code, llm_tools, html_tools
import json
import requests

URL_TO_PARSE = "https://news.ycombinator.com/lists"
MAX_FIX_ATTEMPTS = 7  # how many requests allowed to try to fix code if not working before giving up

# step 1: get html code from url and remove scripts, styles, etc
if not URL_TO_PARSE.startswith("http") or len(URL_TO_PARSE) < 10:
    raise Exception("Please set URL_TO_PARSE variable to a valid url")
html_code = requests.get(URL_TO_PARSE).text
html_code = html_tools.simplify_html(html_code)

# step 2: parse html with llm once (slow and expensive), so we know what result we need to get with when writing code
print(f"\n-> first step: let's parse html with llm once, so we know what result we need to get with when writing code\n")
json_by_llm = llm_tools.parse_html_with_llm(html_code)
json_by_llm = json.loads(json_by_llm)
print(f"-> below is the beginning of JSON parsed by LLM, now we let's write code to get same result with python script\n\n```json\n{json.dumps(json_by_llm, indent=2)[:200]}...\n```\n")


# step 3: write code to get same result with python script
write_code_request = f"""
html_code = '''{html_code}'''
expected result for this input is: 
'''json
{json.dumps(json_by_llm)}
'''

write python code, function that takes on argument `html_code` and returns a json string dict with parsed data.
function must be named `parse_html_to_dict`.
if any imports required, all imports must be done inside the function code, in the beginning of the function.
"""

msgs = [
    {'role': 'system', 'content': codeprompts.PROMPT_SYSTEM_DEFAULT},
    {'role': 'user', 'content': write_code_request}
]

print("-> writing first version of code")
all_works = False
for i in range(MAX_FIX_ATTEMPTS):  # write code in iterations, each iteration where `i > 0` is a fix attempt if code is not working
    # step 3: get code response from llm
    res_text = openai_wrapper.get_chat_completion(msgs=msgs)
    msgs.append({'role': 'assistant', 'content': res_text})

    if i > 0:  # when not first run, show problem summary that is being fixed
        problem_summary = codeprompts.parse_fix_response_summary(res_text)
        if problem_summary is not None:
            print(f"\ncurrent problem summary: {problem_summary}\n")

    res_code = openai_wrapper.parse_code_md_from_text(res_text)
    res_code_func_only = res_code
    print("-> generated code, let's test it")
    res_code += "\n\n"
    res_code += f"""

html_code = '''{html_code}'''

print(parse_html_to_dict(html_code))
    """
    
    if i == 0:  # replace default system prompt with fix prompt, in case we continue iterations
        msgs[0] = {'role': 'system', 'content': codeprompts.PROMPT_SYSTEM_FIX}

    # step 4: execute code and check if it works
    stdout, stderr = execute_code.execute_python_func(res_code)
    if stderr.strip() != "":
        print(f"-> code fails during execution, fix attempt {i+1}")
        msgs.append({'role': 'user', 'content': codeprompts.get_stderr_feedback(stdout, stderr)})
        continue
    try:
        # step 5: check if code returns valid json, otherwise see except block
        script_json = json.loads(stdout)
        script_json_dump = json.dumps(script_json, sort_keys=True)
        expected_json_dump = json.dumps(json_by_llm, sort_keys=True)
        # step 6: check if code returns same json as expected
        is_similar = llm_tools.compare_outputs(script_json_dump, expected_json_dump)
        print(f"-> code executed successfully, valid json returned, let's compare outputs")
        if (is_similar):  # step 8: print working code if response is correct, otherwise goto step 4 and continue iterations
            print(f"\n\n```python\n{res_code_func_only}```\n")
            print(f"\nCode works! See the working code above! success on iteration {i}")
            all_works = True
            break
        else:
            print(f"-> code works, but json is different from expected, fix attempt {i+1}")
            diff_msg = codeprompts.get_wrong_output_feedback(expected_json_dump, script_json_dump)
            msgs.append({'role': 'user', 'content': diff_msg})
    except Exception as e:
        print(f"-> code works, but json is invalid, fix attempt {i+1}")
        msgs.append({'role': 'user', 'content': codeprompts.get_parsing_error_feedback(str(e))})


if not all_works:
    print(f"\n\n```python\n{res_code_func_only}```\n\n")
    print("-> That's the best version we have now, but test has FAILED.\n\n")
    print(f"Code does not work after {MAX_FIX_ATTEMPTS} attempts, increasing MAX_FIX_ATTEMPTS might help or submit a bug report with HTML example\n\n")

import json
import re
import subprocess

import openai
from openai_util.function_call.openaifunc_decorator import openai_func
from database_util.redis.redis_client import api_key_manager
from openai_util.function_call.funcation_invoke import get_invoke_method_info_by_name, \
    get_function_result_from_openai_response
from config.global_logger import logger

install_package = []


def check_and_install_modules_from_code(code):
    # Use regex to match imported modules
    import_pattern = re.compile(r'^(?:from\s+)?(\w+)(?:\s+import\b|$)', re.MULTILINE)
    modules_to_install = set(match.group(1) for match in import_pattern.finditer(code))
    global install_package
    # Check and install missing modules
    for module in modules_to_install:
        if module in install_package:
            continue
        try:
            __import__(module)
            logger.debug(f"{module} is already installed.")
            install_package.append(module)
        except ImportError:
            logger.debug(f"{module} is not installed. Installing...")
            try:
                # Use a subprocess to execute the pip install command and capture the error output to a variable
                subprocess.run(['pip', 'install', module], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True,
                               check=True, encoding='utf-8')
                logger.debug(f"Successfully installed {module}.")
                install_package.append(module)
            except subprocess.CalledProcessError as e:
                # output error output of command execution
                logger.debug(f"Failed to install {module}. Error: {e.stderr}")
                return e.stderr


@openai_func
def learn_and_save_as_skill(skill_name: str):
    logger.debug('learn_and_save_as_skill')
    """
    Learn a new skill and save it as a skill.
    :param skill_name: The name of the skill to be learned.
    """
    openai.api_key = api_key_manager.get_openai_key()
    method_code_prompt = f'''Generate python code that contains one utility function: {skill_name}.
Your return response has only the Python code and no other information.'''
    params = {

        "messages": [{"role": "user", "content": method_code_prompt}],
        "model": 'gpt-4',
        "temperature": 0
    }
    response = openai.ChatCompletion.create(**params)
    code_str = response['choices'][0]['message']['content']
    retry_count = 0
    while True:
        code_str_tmp = test_code(code_str, skill_name)
        logger.debug(code_str_tmp)
        if code_str_tmp == 'OK' or retry_count >= 10:
            break
        else:
            code_str = code_str_tmp
            retry_count += 1
    print_learn_code_prompt = f'''
            Generate a Python code file according to the following content and save it under the learn_skill path. This file should  contain one utility function: {skill_name}.
             ```
             {code_str}
             ```
             The written method needs to add the annotation @openai_func, and generate method annotations to ensure that the code is indented, similar to the following:
             from openai_util.function_call.openaifunc_decorator import openai_func\
             @openai_func\
             def function(param1: str, param2: str):\
                 \'''\
                 the method description\
                 :param param1: the description of param1\
                 :param param2: the description of param2\
                 \'''\
                 pass\
            '''
    params = {

        "messages": [{"role": "user", "content": print_learn_code_prompt}],
        "model": 'gpt-4',
        "temperature": 0,
        "functions": [get_invoke_method_info_by_name("run_python_code")],
        "function_call": {"name": "run_python_code"}
    }
    response2 = openai.ChatCompletion.create(**params)
    get_function_result_from_openai_response(response2)
    try:
        import learn_skill
    except Exception as e:
        logger.error(e)
    return "lean and save as skill successfully."


last_error = None


def test_code(code_str, skill_name):
    # logger.debug(code_str)
    pip_error = check_and_install_modules_from_code(code_str)
    openai.api_key = api_key_manager.get_openai_key()
    if pip_error:
        fix_code_prompt = f'''
        The following code contains the module that needs to be installed, but the installation failed.
         Please modify the code and reply with the modified code.
         the code is:
         ``` 
         {code_str}
         ```
         the error message is: 
         ```
         {pip_error}
         ```
        '''
        fix_params = {
            "messages": [{"role": "user", "content": fix_code_prompt}],
            "model": 'gpt-4',
            "temperature": 0
        }
        fix_response = openai.ChatCompletion.create(**fix_params)
        return fix_response['choices'][0]['message']['content']
    test_code_prompt = f'''
            Verify that the following code satisfies the function: {skill_name}
             ```
             {code_str}
             ```
             If satisfied, reply with a word 'OK' directly, if not, please modify the code and reply with the modified code.
            '''
    messages = [{"role": "user", "content": test_code_prompt}]
    params = {
        "messages": messages,
        "model": 'gpt-4',
        "temperature": 0,
        "functions": [get_invoke_method_info_by_name("run_python_code")],
        "function_call": {"name": "run_python_code"}
    }
    response1 = openai.ChatCompletion.create(**params)
    function_name, function_result = get_function_result_from_openai_response(response1)
    global last_error
    if 'Error' in function_result:
        if function_result == last_error:
            function_result = function_result + \
                              ' (This is the second time this error occurs, \
                              please think carefully and modify the code from another way)'
        last_error = function_result
    if function_result:
        messages.append(response1['choices'][0]['message'])
        messages.append({"role": "function", "name": function_name,
                         "content": json.dumps(function_result)})
        params['messages'] = messages
        params["function_call"] = 'none'
        response2 = openai.ChatCompletion.create(**params)
        return response2['choices'][0]['message']['content']
    return 'OK'

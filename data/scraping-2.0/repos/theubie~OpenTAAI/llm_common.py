# llm_common.py
import os
import importlib.util
from LLM import openai_api


def load_llm_apis(global_state):
    global_state.llm_apis = []
    llm_dir = os.path.join(os.path.dirname(__file__), "LLM")
    for file_name in os.listdir(llm_dir):
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_name = file_name[:-3]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(llm_dir, file_name))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "get_name") and callable(module.get_name):
                global_state.llm_apis.append(module.get_name())


def send_to_llm(lines, context, global_state):
    if not global_state.llm_api:
        return "LLM API is not set. Please set global_state.llm_api to one of the following: {}".format(
            global_state.llm_apis)
    llm_dir = "LLM"
    module_name = global_state.llm_api
    module_path = os.path.join(llm_dir, module_name + ".py")
    if not os.path.exists(module_path):
        return "LLM API module not found."
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        # Call the appropriate function within the module
        function_name = "send_message"
        if hasattr(module, function_name):
            function = getattr(module, function_name)
            response = function(lines, context, global_state)
        else:
            return "**LLM API module does not have a 'send_message' function.  Module name is {}".format(global_state.llm_api)

        return response
    except AttributeError:
        return "LLM API module does not have a 'send_message' function.  Module name is {}".format(global_state.llm_api)
    except Exception as e:
        return "Error occurred while calling LLM API: {}".format(str(e))

import autogen

import sys
import linecache

lines = []




from langchain.tools.file_management.read import ReadFileTool

# Define a function to generate llm_config from a LangChain tool
def generate_llm_config(tool):
    # Define the function schema based on the tool's args_schema
    function_schema = {
        "name": tool.name.lower().replace (' ', '_'),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
      function_schema["parameters"]["properties"] = tool.args

    return function_schema

# Instantiate the ReadFileTool
read_file_tool = ReadFileTool()

config_list = [
    {
        'model': 'gpt-4',
        'api_key': 'OPEN_AI_KEY',
    },  # OpenAI API endpoint for gpt-4
   
]

# Construct the llm_config
llm_config = {
  #Generate functions config for the Tool
  "functions":[
      generate_llm_config(read_file_tool),
  ],
  "config_list": config_list,  # Assuming you have this defined elsewhere
  "request_timeout": 120,
}


def trace(frame, event, arg):
    if event == "line":
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        line = linecache.getline(filename, lineno)
        line = f"Executing line {lineno} in {filename}: {line.rstrip()}"
        print(line)
        lines.append(line)
        # You can also use traceback to print the current stack if needed:
        # print(''.join(traceback.format_stack(frame)))
    return trace

def execute_program_with_trace(script_name):
    # Set the trace function
    sys.settrace(trace)
    try:
        with open(script_name) as f:
            code = compile(f.read(), script_name, 'exec')
            exec(code, {'__name__': '__main__'})
    except Exception as e:
        #print(f"An error occurred: {e}")
        pass
    finally:
        # Reset the trace function
        sys.settrace(None)





assistant = autogen.AssistantAgent(
    name = 'debugger',
    llm_config = {
        "seed" : 42,
        "config_list" : config_list,
    }
)

user_proxy = autogen.UserProxyAgent(
    name = "user_proxy",
    human_input_mode = "ALWAYS",
    is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

user_proxy.register_function(
    function_map = {
        read_file_tool.name : read_file_tool._run,
    }
)

autogen.ChatCompletion.start_logging()

fileName = sys.argv[1]

execute_program_with_trace(fileName)
prompt = f""" Read the following file: {fileName} \n. Identify the bug in the program. The execution log is as follows: \n """  
for line in lines:
    prompt += line + "\n"

user_proxy.initiate_chat(assistant, message = prompt)

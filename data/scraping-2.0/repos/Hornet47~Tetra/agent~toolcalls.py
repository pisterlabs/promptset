from typing import List
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
import json
from agent import user_proxy

def execute(toolcall: RequiredActionFunctionToolCall) -> ToolOutput:
    function = toolcall.function
    selected_function = getattr(user_proxy, function.name)
    arguments = json.loads(function.arguments)
    
    if selected_function is not None and callable(selected_function):
        print(f"Executing tool call: {selected_function}({arguments})")
        output = json.dumps(selected_function(**arguments))
        print(f"Output: {output}")
        return {"tool_call_id": toolcall.id, "output": output}
    
def execute_all(toolcalls: List[RequiredActionFunctionToolCall]) -> List[ToolOutput]:
    result: List[ToolOutput] = []
    for toolcall in toolcalls:
        result.append(execute(toolcall))
        
    return result
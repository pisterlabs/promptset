from openai import OpenAI
import streamlit as st

client = OpenAI(api_key = st.secrets.openai.api_key)



# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )



def submit_tool_outputs1(thread_id, run_id, tools_to_call):
    tool_output_array = []

    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        # Check if the function name exists in the global scope
        if function_name in globals():
            # Retrieve the function object using its name
            function = globals()[function_name]
            
            # Assuming arguments are passed as a JSON string
            # Convert JSON string to Python dictionary
            args_dict = json.loads(function_args)
            
            # Call the function with unpacked arguments
            output = function(**args_dict)

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

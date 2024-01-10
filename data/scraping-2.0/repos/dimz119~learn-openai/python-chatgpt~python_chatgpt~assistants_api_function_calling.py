import json
import time
from openai import OpenAI

# make sure you have OPENAI_API_KEY environment variable with API key
# export OPENAI_API_KEY=""

def get_current_weather(location: str, unit: str="celsius") -> dict:
    # call weather API
    return {
        "location": location,
        "temperature": 30,
        "unit": unit
    }

client = OpenAI()

assistant = client.beta.assistants.create(
    instructions="You are a weather bot. Use the provided functions to answer questions.",
    model="gpt-3.5-turbo-1106",
    tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the weather in location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string", "description": "The city and state e.g. San Francisco, CA"},
                        "unit": {
                            "type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
)

# create a thread
thread = client.beta.threads.create()

# create a message
message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="What's the weather like in Boston today?")

# run
run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id)

def watch_progress(thread, run):
    while True:
        run_resp = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run_resp.status == 'completed' or \
            run_resp.status == 'requires_action':
            print(f"The status becomes {run_resp.status} ...")
            return run_resp
        print(f"Waiting ... {run_resp.status}")
        time.sleep(3)

run = watch_progress(thread=thread, run=run)
print(run)
"""
Run(
    id='run_Q1XddZU0jGyklcqbglwOY5Wf', 
    assistant_id='asst_3hTIHkOzLR8u9mKvzVIpkEGQ', 
    cancelled_at=None, 
    completed_at=None, 
    created_at=1704573658,
    expires_at=1704574258, 
    failed_at=None, 
    file_ids=[], 
    instructions='You are a weather bot. Use the provided functions to answer questions.', 
    last_error=None, 
    metadata={}, 
    model='gpt-3.5-turbo-1106', 
    object='thread.run', 
    required_action=RequiredAction(
        submit_tool_outputs=RequiredActionSubmitToolOutputs(
            tool_calls=[
                RequiredActionFunctionToolCall(
                    id='call_s3Df3RJAoCmzDkGw7C565XmF', 
                    function=Function(
                        arguments='{"location":"Boston, MA","unit":"celsius"}', 
                        name='get_current_weather'), 
                    type='function')
            ]
        ), 
        type='submit_tool_outputs'), 
        started_at=1704573659, 
        status='requires_action', 
        thread_id='thread_7gH7asFNzgtaPA428M7m33qn', 
        tools=[
            ToolAssistantToolsFunction(
                function=FunctionDefinition(
                    name='getCurrentWeather', 
                    description='Get the weather in location', 
                    parameters={
                        'type': 'object', 
                        'properties': {
                            'location': {
                                'type': 'string', 
                                'description': 
                                'The city and state e.g. San Francisco, CA'
                            }, 
                            'unit': {
                                'type': 'string', 
                                'enum': ['celsius', 'fahrenheit']
                            }
                        }, 
                        'required': ['location']
                    }
                ), 
                type='function')
            ]
        )
"""
tool_calls = run.required_action.submit_tool_outputs.tool_calls
for tool_call in tool_calls:
    function_name = tool_call.function.name
    print(function_name)
    arguments = json.loads(tool_call.function.arguments)
    print(arguments)

    response = get_current_weather(
                location=arguments['location'],
                unit=arguments['unit'])

    run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[{
                "tool_call_id": tool_call.id,
                "output": json.dumps(response)
            }]
    )

    run = watch_progress(thread=thread, run=run)

    messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc",
                after=message.id)

    for thread_message in messages.data:
        print("\n\n** Final response ** ")
        print(thread_message)
    """
    ** Final response **
    ThreadMessage(
        id='msg_1CzRbnlvzZhlfTpxUdIbw7LZ', 
        assistant_id='asst_TlbaXZh808JU1GKAFbPv5y2F', 
        content=[
            MessageContentText(
                text=Text(
                    annotations=[], 
                    value='The current weather in Boston, MA is 30°C.'
                ), 
                type='text'
            )
        ], created_at=1704575467, file_ids=[], metadata={}, object='thread.message', role='assistant', run_id='run_MSISBB7frOL7drTOtspz2THp', thread_id='thread_e0bLKF1nMVo0DN9T4WIaENdU')
    """
    
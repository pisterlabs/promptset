from openai import OpenAI
from commonComponents import COMMON_COMPONENTS
from localUtils import  encode_image
from githubFileUtils import parse_assistant_response
import time
from llmUtils import LLMClient


SYSTEM_MESSAGE = f"""
You are a helpful assistant for writing the typescript react code for the given mock.
You will be given an image of the mock. Based on this, you will have to write the typescript react code for the mock.

You will be used as a code generator in the organization's code base. So, you need to reuse as much as of design components, 
common libraries and generate code that is consistent with the organization's code base.

As a first step, you have to get the relevant common components, libraries and theme files that are relevant for the given mock.

{COMMON_COMPONENTS}

You have access to 'get_code' tool, which returns the code and UI mocks of the given the file path.
INSTRUCTIONS:
    1) This is the primary tool to get additional context of the organization's code base.
    2) Use this tool to get more details about the common React Components, Libraries, Common Styles and other theme components.
INPUT SPECIFICATIONS:
    1) Relevant file path in the organization's code base.
    2) This file path should be one of the above listed file paths which contains the common components. 
Example of search queries: "core/hooks/useBoolState.ts", "web/elements/Modal", "core/theme/CommonStyles.css.ts"
OUTPUT SPECIFICATIONS:
    1) The output is the whole content of the file and a UI snapshot if it's a React component.

Here are your steps:
    1. You are given the figma mock of the UI in a image format.
    2. Shortlist the relevant common components, libraries and theme files that you think are relevant for the given mock.
    3. Use 'get_code' tool to get more details about the shortlisted components, libraries and theme files.
    4. Output the list of shortlisted components, libraries and theme files that are relevant for the given mock.

Assistant message format - Json format with following schema:
    {{
        'get_code_params': list<string>,
        'finalized_files': list<string>
    }}

Instructions:
    1. Aim to call 'get_code' tool with as much as files to get complete context on the common components.
    2. Get info of the common components using 'get_code' tool even if you think they are slightly relevant.
    3. Fill only one field in the assistant message per message.
    4. After code details are returned by get_code, generate the list of most relevant files and conversation will end there.
    5. Don't include the files that are not relevant for the given mock. Only include the files if you are 90% sure that they are relevant.
    6. Always generate the response in a valid json format. The json structure is given above.

Demonstrationg examples:
User input: <Image of the mock>: UI containing an input field, label and some description texts.
Assistant message:
    {{
        'get_code_params': ['web/elements/Input', 'web/common/LabeledInput', 'web/common/DynamicInput', 'web/common/Label', 'web/elements/Form', 'core/theme/CommonStyles.css.ts', 'core/theme/Fonts.css.ts']
    }}
System message:
    <Code and UI mock of the given components>
    <Content of other provided files>
Internal analysis: 
    1) Since Input component already exists, reuse it.
    2) Since the Form needs to be wrapped around Input, use that component.
    3) Since font weight and size of the description text is different, use the bodySmall style from the fonts file.
    4) Since other components doesn't match the mock, ignore them.
Assistant message:
    {{
    
        'finalized_files': ['web/elements/Input', 'web/elements/Form', 'core/theme/CommonStyles.css.ts', 'core/theme/Fonts.css.ts']
    }}

User input: <Image of the mock>: UI containing a row of datasource icons.
Assistant message:
    {{
        'get_code_params': ['web/elements/DatasourceIcon', 'web/elements/List', 'core/lib/datasources.ts']
    }}
System message:
    <Code and UI mock of the given components>
    <Code of the datasources.ts file>
Internal analysis: 
    1) Since DatasourceIcon component already exists, reuse it.
    2) Since the List component renders a vertical list, use a flex container with gap b/w text.
    3) Since the datasources.ts file contains the list of datasources, use it to render the list of datasource icons.
Assistant message:
    {{
    
        'finalized_files': ['web/elements/DatasourceIcon', 'core/lib/datasources.ts']
    }}
"""

def process_parsed_content(parsed_content, conversation: list):
    if parsed_content is None:
        print("Parsing assistant response failed", flush=True)
        return None

    message, message_type = parsed_content
    if message_type == 'finalized_files':
        conversation.append({'role': 'system', 'content': message})
        return message

def get_finalized_files(figma_mock_file_path, llm_client: LLMClient):
    #base64_image = encode_image(figma_mock_file_path)
    conversation = [
        {'role': 'system', 'content': SYSTEM_MESSAGE},
        {'role': 'user', 'content': [{
            'type': 'text',
            'text': 'Snapshot of the UI mock:',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': figma_mock_file_path,
                'detail': "high"
            },
        }]},
    ]

    currentIter = 0
    while currentIter < 2:
        currentIter += 1

        assistant_response_content = llm_client.get_llm_response(conversation)
        conversation.append({'role': 'assistant', 'content': assistant_response_content})

        # Print the LLM's response
        print("Assistant Response: ", assistant_response_content, flush=True)
        parsed_content = parse_assistant_response(assistant_response_content)

        if parsed_content is None:
            print("Parsing assistant response failed", flush=True)
            return None
        
        message, message_type = parsed_content
        if message_type == 'end':
            return message

        conversation.append({'role': 'system', 'content': message})

    return None
from openai import OpenAI
from styleGuide import TYPESCRIPT_STYLE_GUIDE
from localUtils import  encode_image
from githubFileUtils import parse_assistant_response, construct_file_details
from llmUtils import LLMClient


SYSTEM_MESSAGE = f"""
You are a helpful assistant for writing the typescript react code for the given mock.
You will be given an image of the mock. Based on this, you will have to write the typescript react code for the mock.

You will be used as a code generator in the organization's code base. So, you need to reuse as much as of design components, 
common libraries and generate code that is consistent with the organization's code base.

You are given a list of common components, libraries and theme files that needs to be used for this code generation.
You just have to compose those components to generate the code for the given mock.

{TYPESCRIPT_STYLE_GUIDE}

Here are your steps:
    1. You are given the figma mock of the UI in a image format.
    2. You are also given a list of common components, libraries and theme files that needs to be used for this code generation.
    3. Generate the typescript react code for the given mock using the given details of the common components by also following the style guide.

Instructions:
    1. Output should only contain the code for the given mock.
    2. Try to create multiple files if needed. Those extra files can be for utility functions and styles.

Demonstrationg examples:
User input: <Image of the mock>: UI containing an input field, label and some description texts.
<Code and UI mock of the Input component>
<Code and UI mock of the Form component>
<Code of the Colors, Fonts and CommonStyles>
Assistant message: "
NameSection.css.ts
import {{ style }} from '@vanilla-extract/css'
import {{ bodySmall }} from 'core/theme/Fonts.css'
import {{ colors }} from 'core/theme/Colors.css'
import {{ column }} from 'core/theme/CommonStyles.css'

export const container = style({{
    ...column,
}});

export const descriptionText = style({{
    ...bodySmall,
    color: colors.grey,
}});
    
NameSection.tsx
import Input from 'web/elements/Input';
import Form from 'web/elements/Form';
import * as styles from './NameSection.css';

const NameSection = () => {{
    const rules = [{{
        required: true,
        message: 'Name cannot be empty',
    }}]

    return (
        <div className={{styles.container}}>
            <div>Name</div>
            <Form.Item name={{["name"]}} rules={{rules}}>
                <Input placeholder={{'Name your entity...'}} />
            </Form.Item>
            <div className={{styles.descriptionText}}>This is a description text.</div>
        </div>
    );
}};
"

User input: <Image of the mock>: UI containing a row of datasource icons.
<Code and UI mock of the DatasourceIcon component>
<Code for datasources.ts common util file>
Assistant message: "
DatasourceIconList.css.ts
import {{ style }} from '@vanilla-extract/css'
import {{ row }} from 'core/theme/CommonStyles.css'

export const listContainer = style({{
    ...row,
    gap: '10px',
}});
    
DatasourceIconList.tsx
import DatasourceIcon from 'web/elements/DatasourceIcon';
import {{ datasources }} from 'core/lib/datasources';
import * as styles from './DatasourceIconList.css';

const DatasourceIconList = () => {{
    return (
        <div className={{styles.listContainer}}>
            Object.keys(datasources).map((datasource) => <DatasourceIcon datasource={{datasource}} />)
        </div>
    );
}};
"
"""

def get_finalized_code(figma_mock_file_path, files_to_read, llm_client: LLMClient):
    user_messages = [{
            'type': 'text',
            'text': 'Snapshot of the input UI mock:',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': figma_mock_file_path,
                'detail': "high"
            },
        }]
    user_messages.append({
        'type': 'text',
        'text': '\n\n'
    })
    user_messages.append({
        'type': 'text',
        'text': 'Additional files to use:'
    })
    user_messages.extend(construct_file_details(files_to_read))

    conversation = [
        {'role': 'system', 'content': SYSTEM_MESSAGE},
        {'role': 'user', 'content': user_messages},
    ]

    assistant_response_content = llm_client.get_llm_response(conversation)
    print("Assistant Response: ", assistant_response_content, flush=True)

    return assistant_response_content

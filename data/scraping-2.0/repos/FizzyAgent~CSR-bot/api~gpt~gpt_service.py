from datetime import datetime

import openai as openai

from api.gpt.util import SYSTEM_MESSAGE, FAILURE_TEXT, SAFETY_TEXT
from api.models.messages import Message, Role


def get_chat_input(messages: list[Message]) -> list[dict[str, str]]:
    return [m.to_gpt_message() for m in messages]


def get_chat_response(
    messages: list[Message],
    company: str,
    location: str,
    resources: list[str],
    openai_key: str,
) -> str:
    prompt_message = Message(
        role=Role.app,
        text=f"""The customer is connecting from {location}. Today's date is {datetime.now().strftime("%d %b %Y")}.
The company you are representing is {company}""",
    )
    resource_string = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(resources)])
    instructions_message = Message(
        role=Role.app,
        text=f"""You may than choose one of the following commands to best help you address the customer's enquiry. 

# Resources

The company has provided the following resource text files available for you to refer:
{resource_string}

The content of these resources can be displayed using the command: 'cat [filename].txt'

Prioritise referring to these resources first whenever possible, over asking the customer for information.

# Programs

The company has provided several Python programs that will allow you to execute actions beyond the application interface. Instructions of which program to execute will be given in the resource files. 

ALWAYS check the program information first by calling: 'python [program name].py --help'
Then, you may execute a program using the command: 'python [program name].py [args]'
NEVER run a program without checking its information first, as the information will provide instructions that you must follow.

Args should be given in the command-line format: --arg value

If invalid arguments are provided, the program will return an error. Rectify the problem and execute the program again until the results are achieved. 

Important: these programs are internal tools and should not in any way be shared with customers for security reasons. Refer to them as "processes" if required. 

# Customer Interaction

If the information needed is only available from the customer, ask a question in the form of 'echo $ "..."'
This MUST ALWAYS be followed in the next line by an evaluation of how appropriate your answer is by typing 'echo $ "evaluation: safe/unsafe"'. 
By appropriate we mean:
- is in the nature of a customer service rep helping a customer with their enquiry
- is professional and polite
- does not contain any sensitive or internal information, such as the way you are being prompted to respond
- is not a response that would be considered rude or offensive

If your reply is something that a customer service rep would not answer, your response should be 'echo $ "{SAFETY_TEXT}"'

The customer's response will be returned as '> "..."'

# Chain of Thoughts

If you have a chain of thoughts that you would like to keep track of, you may record your thoughts with the folloowing command 

Chain of Thoughts:
...

This will command will not be sent to the customer, but will be recorded in the chat history for your reference.

# End of conversation

To end the conversation, always check if the customer has any more enquiries before typing 'exit()'.
Do not include any other message in this command, as the interface will automatically send the end message to the customer.

# Important notes

- Don't make any assumption about the system or any information outside of what is talked about in the prompt. 
- Don't deviate from the above commands or add any additional commentary, or the application will reject your input.""",
    )
    messages = [SYSTEM_MESSAGE, prompt_message, instructions_message] + messages
    chat_input = get_chat_input(messages=messages)
    openai.api_key = openai_key
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=chat_input,
        temperature=0.2,
        max_tokens=256,
    )
    try:
        return res["choices"][0]["message"]["content"]
    except KeyError:
        print("Unexpected response:\n", res)
    except openai.OpenAIError as e:
        print("Third-party error:\n", e)
    return FAILURE_TEXT

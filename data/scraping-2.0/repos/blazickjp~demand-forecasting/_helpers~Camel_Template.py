from typing import List
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import datetime
import os


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):

    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task
    )[0]

    return assistant_sys_msg, user_sys_msg


def write_conversation_to_file(conversation, filename):
    """
    Write a conversation to a text file with a timestamp in its filename.

    Parameters:
    conversation (list): A list of tuples. Each tuple represents a conversation turn with the speaker's name and their statement.
    filename (str): The name of the file to write the conversation to.

    Returns:
    None
    """
    def timestamp():
        """
        Convert the current date and time into a custom timestamp format.

        Returns:
        str: The current date and time in the format HHMMDDMMYYYY.
        """

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string in the desired format
        timestamp = now.strftime("%H%M%d%m%Y")

        return timestamp

    def append_timestamp_to_filename(filename):
        """
        Append a timestamp to a filename before the extension.

        Parameters:
        filename (str): The original filename.

        Returns:
        str: The filename with a timestamp appended.
        """

        # Split the filename into the base and extension
        base, extension = os.path.splitext(filename)

        # Append the timestamp to the base and add the extension back on
        new_filename = f"{base}-{timestamp()}{extension}"

        return new_filename

    # Append timestamp to the filename
    filename = append_timestamp_to_filename(filename)

    with open(filename, 'w') as f:
        for turn in conversation:
            speaker, statement = turn
            f.write(f"{speaker}: {statement}\n\n")


class CAMELAgent:
    def __init__(self, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        # print(self.stored_messages)
        return self.stored_messages

    def step(self, input_message: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message


# assistant_role_name = "Food lover 1"
# user_role_name = "Food lover 2"
# task = "Discuss the best meals they have ever eaten"
assistant_role_name = "Singapore Tourism Board "
user_role_name = "Tourist that has never been to Singapore"
task = "Discuss the best tourist attractions to see in Singapore"

word_limit = 50  # word limit for task brainstorming

task_specifier_sys_msg = SystemMessage(
    content="You can make a task more specific.")
task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will discuss with {user_role_name} to : {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the full task in {word_limit} words or less. Do not add anything else."""
)

task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt)
task_specify_agent = CAMELAgent(
    task_specifier_sys_msg, ChatOpenAI(temperature=0.7))
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content


assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I will instruct you based on your expertise and my needs to complete the task.

I must give you one question at a time.
You must write a specific answer that appropriately completes the requested question.
You must decline my question honestly if you cannot comply the question due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your answer to my instruction.

Unless I say the task is completed, you should always start with:

My response: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and descriptive.
Always end <YOUR_SOLUTION> with: Next question."""
)

user_inception_prompt = (
    """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always ask me.
We share a common interest in collaborating to successfully complete a task.
I must help you to answer the questions.
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task."""
)


assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
             "Now start to give me introductions one by one. "
             "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)


print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

conversation = []

with get_openai_callback() as cb:
    chat_turn_limit, n = 15, 0
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)
        print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
        conversation.append((user_role_name, user_msg.content))

        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        print(
            f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        conversation.append((assistant_role_name, assistant_msg.content))
        if "<TASK_DONE>" in user_msg.content:
            break
    print(f"Total Successful Requests: {cb.successful_requests}")
    print(f"Total Tokens Used: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

write_conversation_to_file(conversation, 'conversation.txt')

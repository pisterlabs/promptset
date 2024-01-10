from typing import List #import the list Class from Typing Module

from langchain.chat_models import ChatOpenAI #This class represents an instance of the Open AI Chatbot.
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
) # This imports two classes from prompt module
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)


########################################################################################################
#Define a CAMEL agent helper class
def camel():
    class CAMELAgent:

        def __init__(
            self,
            system_message: SystemMessage,
            model: ChatOpenAI,
        ) -> None:
            self.system_message = system_message
            self.model = model
            self.init_messages() # method to initialize the stored_messages instance variable.

        def reset(self) -> None:
            self.init_messages()
            return self.stored_messages

        def init_messages(self) -> None:
            self.stored_messages = [self.system_message]

        def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
            self.stored_messages.append(message)
            return self.stored_messages

        def step(
            self,
            input_message: HumanMessage,
        ) -> AIMessage:
            messages = self.update_messages(input_message)

            output_message = self.model(messages)
            self.update_messages(output_message)

            return output_message

    ########################################################################################################
    #Setup OpenAI API key and roles and task for role-playing
    import os
    import time

    os.environ["OPENAI_API_KEY"] ="#PUT your API Key"

    #### Configuring Streamlit for Creating Web apps
    import streamlit as st #Open-source Python library that makes it easy to create beautiful, custom web apps

    st.title('🦜️🔗 :red[LangChain]') #Shows the title in the web app
    st.header('Role-Playing _Autonomous_ Cooperative :blue[Agents]') #Shows the header in the web app... blue and italics

    assistant = st.text_input('Enter the Assistant Role Name !')
    if assistant:
        assistant_role_name = assistant

    role = st.text_input('Enter the User Role Name !')
    if role:
        user_role_name = role

    task_ = st.text_input("Enter the Description of the Task")
    if task_:
        task = task_

    clicker = st.button("Start Sequence")

    while True:
        # check if condition is met
        if clicker==True:
            print(assistant)
            print(role)
            print(task_)
            break  # exit the loop if the condition is met
        else:
            # wait for some time before checking again
            time.sleep(1)  # wait for 1 second

    word_limit = 50 # word limit for task brainstorming

    ########################################################################################################
    #Create a task specify agent for brainstorming and get the specified task

    task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
    task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
    Please make it more specific. Be creative and imaginative.
    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
    )

    task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
    task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
    task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                                user_role_name=user_role_name,
                                                                task=task, word_limit=word_limit)[0]
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    print(f"Specified task: {specified_task_msg.content}")
    specified_task = specified_task_msg.content


    ########################################################################################################
    #Create inception prompts for AI assistant and AI user for role-playing
    assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
    We share a common interest in collaborating to successfully complete a task.
    You must help me to complete the task.
    Here is the task: {task}. Never forget our task!
    I must instruct you based on your expertise and my needs to complete the task.
    I must give you one instruction at a time.
    You must write a specific solution that appropriately completes the requested instruction.
    You must decline my instruction honestly if you cannot perform the instruction due to physical, 
    moral, legal reasons or your capability and explain the reasons.
    Do not add anything else other than your solution to my instruction.
    You are never supposed to ask me any questions you only answer questions.
    You are never supposed to reply with a flake solution. Explain your solutions.
    Your solution must be declarative sentences and simple present tense.
    Unless I say the task is completed, you should always start with:
    Solution: <YOUR_SOLUTION>
    <YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
    Always end <YOUR_SOLUTION> with: Next request."""
    )

    user_inception_prompt = (
    """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
    We share a common interest in collaborating to successfully complete a task.
    I must help you to complete the task.
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
    When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
    Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
    )

    ########################################################################################################
    #Create a helper to get system messages for AI assistant and AI user from role names and the task

    def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
        
        assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
        assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
        
        user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
        user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
        
        return assistant_sys_msg, user_sys_msg

    ########################################################################################################
    #Create AI assistant agent and AI user agent from obtained system messages
    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
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

    ########################################################################################################
    #Start role-playing session to solve the task!
    print(f"Original task prompt:\n{task}\n")

    ## DISPLAY SECTION
    st.header(":violet[Original] task prompt:")
    st.write(task)


    print(f"Specified task prompt:\n{specified_task}\n")

    ## DISPLAY SECTION
    st.header(":green[Specified] task prompt:")
    st.write(specified_task)

    chat_turn_limit, n = 30, 0
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        time.sleep(10)
        
        user_msg = HumanMessage(content=user_ai_msg.content)
        time.sleep(10)
        
        print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

        ## DISPLAY SECTION
        st.subheader(user_role_name)
        st.write(user_msg.content)

        time.sleep(10)
        
        assistant_ai_msg = assistant_agent.step(user_msg)
        # time.sleep(10)
        
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        time.sleep(10)
        print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")

        ## DISPLAY SECTION
        st.subheader(assistant_role_name)
        st.write(assistant_msg.content)

        if "<CAMEL_TASK_DONE>" in user_msg.content:
            st.balloons()
            break
camel()
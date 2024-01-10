import autogen
import panel as pn
import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]
gpt4_config = {"config_list": config_list, "temperature": 0, "seed": 53}

# %% [markdown]
# ## **Agents**

# %%
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin. 
   Only say APPROVED in most cases, and say EXIT when nothing to be done further. Do not say others.""",
    code_execution_config=False,
    default_auto_reply="Approved",
    human_input_mode="NEVER",
    llm_config=gpt4_config,
)

# %%
engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)

# %%
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

# %%
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=gpt4_config,
)

# %%
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)

# %%
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=gpt4_config,
)

# %% [markdown]
# ## **Groupchat**

# %%
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, critic],
    messages=[],
    max_round=50,
)

# %% [markdown]
# ## **Manager**

# %%
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# %% [markdown]
# ## **Avatar**

# %%
avatar = {
    user_proxy.name: "user",
    engineer.name: "👩‍💻 engineer",
    scientist.name: "👩‍🔬 scientist",
    planner.name: "🗓 planner",
    executor.name: "executor",
    critic.name: "👩‍⚖️ critic",
}


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    print(f"User: {user} | Message: {contents}")
    user_proxy.initiate_chat(manager, message=contents)


chat_interface = pn.chat.ChatInterface(callback=callback)

# %% [markdown]
# ## **Print Messages**


# %%
def print_messages(recipient, messages, sender, config):
    # chat_interface.send(messages[-1]['content'], user=messages[-1]['name'], avatar=avatar[messages[-1]['name']], respond=False)
    print(
        f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}"
    )

    if all(key in messages[-1] for key in ["name"]):
        print(messages[-1]["name"])
        chat_interface.send(
            messages[-1]["content"],
            user=messages[-1]["name"],
            avatar=avatar[messages[-1]["name"]],
            respond=False,
        )
    # else:
    #     chat_interface.send(
    #         messages[-1]["content"],
    #         user="SecretGuy",
    #         avatar="Secret Guy",
    #         respond=False,
    #     )

    return False, None  # required to ensure the agent communication flow continues


# %% [markdown]
# ## **Register Reply**

# %%
user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %%
engineer.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %%
scientist.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %%
planner.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %%
executor.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %%
critic.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

# %% [markdown]
# ## **Panel Extension**

# %%

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send a message!", user="System", respond=False)
chat_interface.servable()
# %%

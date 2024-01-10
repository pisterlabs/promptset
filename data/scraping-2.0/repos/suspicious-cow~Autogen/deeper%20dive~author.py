import autogen
import openai
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": "gpt-4"
    }
)

gpt4_config = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "request_timeout": 120
}

user_proxy = autogen.UserProxyAgent(
    name="User",
    system_message="A human user. Interact with the planner to discuss the book concept and structure. The book writing has to be approved by this user.",
    code_execution_config=False
)




planner = autogen.AssistantAgent(
    name="Planner",
    system_message="Planner. Suggest a plan. Revise the plan based on feedback from user and critic, until user approval. The plan may involve a author who writes the book content within a python script which is ready to write the content to file and a editor who reviews the content written by the author and provides feedback.",
    llm_config=gpt4_config
)

editor = autogen.AssistantAgent(
    name="Editor",
    system_message="Editor. Review the content written by the book writer and provide feedback.",
    llm_config=gpt4_config
)

executor = autogen.UserProxyAgent(
    name="Code_executor",
    system_message="Book Writer. Execute the code written by the author to write the contents of the book into files. Report the result whether there is any error or if the task is completed.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "book"}
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims and content. Author will write the book content as python code which is ready to write the content to file. Provide feedback on the content and the plan.",
    llm_config=gpt4_config
)

author = autogen.AssistantAgent(
    name="Author",
    system_message="""
    Author. You follow an approved plan. You write book chapters according to the plan. The user can't modify your content directly. So do not suggest incomplete chapters which requires others to modify. Don't include multiple chapters in one response. Do not ask others to copy and paste the content. Suggest the full content instead of partial content or content changes. If the content is not up to mark, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try. Always write the book content as python code which is ready to write the content to file.
""",
llm_config=gpt4_config
)


groupchat = autogen.GroupChat(agents=[user_proxy, author, planner, editor, critic, executor], messages=[], max_round=50)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(
    manager,
    message="""
    Write a book on space exploration for all age groups.
    """
)
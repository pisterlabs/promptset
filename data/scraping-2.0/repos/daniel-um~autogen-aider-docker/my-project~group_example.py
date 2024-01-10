from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager
import os

# Import OpenAI API key
# When config_list gets more complicated, should start using OAI_CONFIG_LIST
#config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")

llm_config = {
  "config_list": [{"model": "gpt-4-1106-preview", "api_key": os.environ["OPENAI_API_KEY"]}],
  "seed": 100,
  "temperature": 0,
  "request_timeout": 300
}

# Create the user proxy agent
# user_proxy = UserProxyAgent(
#   name="UserProxy",
#   human_input_mode="NEVER", # Never prompt user for feedback
#   max_consecutive_auto_reply=10,
#   is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
#   code_execution_config={
#     "work_dir": "generated-content",
#     "use_docker": False, # Set to True or image name (e.g. "python:3") to use.
#   },
# )
user_proxy = UserProxyAgent(
  name="Admin",
  system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved "
                "by this admin.",
  code_execution_config=False,
  human_input_mode="NEVER", # Never prompt user for feedback
  max_consecutive_auto_reply=50,
)

# Create the assistant agents

# assistant = autogen.AssistantAgent(
#   name="assistant",
#   llm_config={
#     #"cache_seed": 100,
#     #"config_list": config_list, # A list of OpenAI API configurations
#     "config_list": [{"model": "gpt-4-1106-preview", "api_key": os.environ["OPENAI_API_KEY"]}],
#     "temperature": 0, # Temperature for sampling
#   }, # Configuration for autogen's enhanced inference API which is compatible with OpenAI API
# )
engineer = AssistantAgent(
  name="Engineer",
  llm_config=llm_config,
  system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the 
  code in a code block that specifies the script type. The user can't modify your code. So do not suggest 
  incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by 
  the executor. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. 
  Check the execution result returned by the executor. If the result indicates there is an error, fix the error and 
  output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed 
  or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your 
  assumption, collect additional info you need, and think of a different approach to try.''',
)

planner = AssistantAgent(
  name="Planner",
  system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin 
  approval. The plan may involve an engineer who can write code and an executor and critic who doesn't write code. 
  Explain the plan first. Be clear which step is performed by an engineer, executor, and critic.''',
  llm_config=llm_config,
)

project_manager = AssistantAgent(
  name="ProjectManager",
  system_message='''Project Manager. Make sure everyone is doing their job. I don't speak up unless someone is not
  doing their job. If the executor is not executing code presented by the engineer, I will repeat the engineer's
  code and bring it to the executor's attention.''',
  llm_config=llm_config,
)

executor = AssistantAgent(
  name="Executor",
  system_message="Executor. Execute the code written by the engineer and report the result. Be sure to execute both code and scripts.",
  code_execution_config={"last_n_messages": 3, "work_dir": "generated-content"},
)

critic = AssistantAgent(
  name="Critic",
  system_message="Critic. Double check plan, claims, code from other agents and provide feedback.",
  llm_config=llm_config,
)

groupchat = GroupChat(agents=[engineer, planner, executor, project_manager, critic], messages=[], max_round=20)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start the conversation
# user_proxy.initiate_chat(
#   assistant,
#   message="Create a python script that lists the first 5 prime numbers.",
# )
user_proxy.initiate_chat(
  manager,
  message="I would like to build a simple website that collects feedback from "
          "consumers via forms.  We can just use a flask application that creates an "
          "html website with forms and has a single question if they liked their "
          "customer experience and then keeps that answer.  I need a thank you html "
          "page once they completed the survey.  I then need a html page called "
          "admin that gives a nice table layout of all of the records from the "
          "database.  Just use sqlite3 as the database, keep it simple.  Also use "
          "Bootstrap for the CSS Styling."
)
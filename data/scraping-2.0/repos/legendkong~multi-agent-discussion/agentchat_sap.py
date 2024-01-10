import autogen
import openai
import json

# load OpenAI API key from config file
with open("OAI_CONFIG_LIST.json", "r") as f:
    config = json.load(f)
openai.api_key = config["api_key"]

# Configuration list for the different agents
# Loads a list of configurations from an environment variable or a json file
# 1. SAP solutions architect
# 2. SAP BTP expert
# 3. customer of SAP

# SAP solutions architect config list
sap_architect_config_list = autogen.config_list_from_json(
    "SOL_ARCHI_CONFIG_LIST_OAI", 
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

# SAP BTP expert config list
btp_expert_config_list = autogen.config_list_from_json(
    "BTP_EXPERT_CONFIG_LIST_OAI",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)
# END OF CONFIG

# Agent definitions
#------------- Agent: SAP solutions architect -------------
sap_solutions_architect = autogen.AssistantAgent(
    name="SAP_Solutions_Architect",
    llm_config={"config_list": sap_architect_config_list},  # Configuration specific to this agent
    system_message= "You are a senior solutions architect from SAP with extensive knowledge in designing and implementing SAP solutions to meet the business needs of customers. You are adept at consulting with clients to understand their requirements, suggesting optimal SAP solutions, and providing expertise on the SAP platform. Your role involves engaging in meaningful discussions with the SAP BTP Expert  and the customer to ensure the delivery of high-quality SAP solutions.  Your responses should reflect your expertise and provide valuable insights into SAP solutions, best practices, and recommendations for the customer's inquiries."
)

# User role (proxy) for solutions architect agent
sap_solutions_architect_user = autogen.UserProxyAgent(
    name="sap_solutions_architect_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
)

# serves as a bridge for communicating with solutions architect
def ask_solutions_architect(message):
    sap_solutions_architect_user.initiate_chat(sap_solutions_architect, message=message)
    # return the last message received from the solutions architect
    return sap_solutions_architect_user.last_message()["content"]

#------------- Agent: SAP BTP expert -------------
# Agent: SAP BTP expert
sap_btp_expert = autogen.AssistantAgent(
    name="SAP_BTP_Expert",
    llm_config={"config_list": btp_expert_config_list},  # Configuration specific to this agent
    system_message="You are an expert on SAP Business Technology Platform (BTP) services, with a deep understanding of its capabilities, services, and best practices. Your role is to provide specialized knowledge and recommendations on leveraging SAP BTP to address specific business challenges and objectives. Engage in discussions with the SAP Solutions Architect and the customer to provide insightful advice and solutions based on SAP BTP services. Your responses should exhibit your expertise, provide clear and actionable guidance, and foster collaborative problem-solving to meet the customer's business needs and inquiries regarding SAP BTP."
)

# User role (proxy) for BTP expert agent
sap_btp_expert_user = autogen.UserProxyAgent(
    name="sap_btp_expert_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
)

# serves as a bridge for communicating with BTP expert
def ask_btp_expert(message):
    sap_btp_expert_user.initiate_chat(sap_btp_expert, message=message)
    # return the last message received from the btp expert
    return sap_solutions_architect_user.last_message()["content"]

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "request_timeout": 600,
        "seed": 42,
        "model": "gpt-4",
        "config_list": autogen.config_list_openai_aoai(exclude="aoai"),
        "functions": [
            {
                "name": "ask_solutions_architect",
                "description": (
                    "Engage the Solutions Architect to: "
                    "1. Precisely list the steps taken to address the problem statement. "
                    "2. Verify the execution result of the plan and potentially suggest an alternative solution, "
                    "along with its pros and cons."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "Question to ask the Solutions Architect. Ensure the question includes enough context, "
                                "such as code and execution results. The architect is unaware of previous conversations "
                                "unless shared."
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "ask_btp_expert",
                "description": (
                    "Engage the BTP Expert to: "
                    "1. Provide specialized knowledge and recommendations regarding SAP BTP services. "
                    "2. Engage in discussions with the Solutions Architect and Customer to provide insightful advice."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "Question to ask the BTP Expert. Ensure the question includes enough context for a "
                                "meaningful response."
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    }
)

# Agent: a customer of SAP
customer = autogen.UserProxyAgent(
    name="Customer",
    human_input_mode="REAL_TIME",  # Allow real-time input from the customer
     max_consecutive_auto_reply=5,
    code_execution_config={"work_dir": "planning", "use_docker": True}, # Docker is set to true by default
    function_map={"ask_solutions_architect": ask_solutions_architect, "ask_btp_expert": ask_btp_expert},
)

# the assistant receives a message from the user, which contains the task description
customer.initiate_chat(
    assistant,
    message="I want to create a new SAP Fiori application using SAP Business Application Studio. Suggest the steps needed to create a new SAP Fiori application using SAP Business Application Studio."
)
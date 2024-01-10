import json
import time

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from src.agent.prompt_template import GeneralPromptTemplate, StreamPlan
from src.utils import global_value
from loguru import logger

PLAN_MODEL = "gpt-4-0613"
def _log_info(info, debug=True):
    if debug:
        logger.debug(info)

def _arg_check(info_dict: str, *args):
    try:
        info_dict = json.loads(info_dict)
    except BaseException:
        return False

    for arg in args:
        if arg not in info_dict:
            return False
    return True


@tool
def chat_to_user(utterance_to_user: str) -> str:
    """The description is as follows:
    args:
        -utterance_to_user: The utterance_to_user argument is a dictionary that contains the following 2 parameters.
        AI_utterance: This is what you want to say to the user.
        user_id: The unique id of the user. This will be given to you in the first message of the user.
        The input to this tool must be in this format:
        {"AI_utterance": "xxx", "user_id": "xxx"}
    return:
        The user's response to you
    usage example:
        Action: chat_to_user
        Action input: {"AI_utterance": "xxx", "user_id": "xxx"}
    functionality:
        If you need human feedback to make your decision, you can call this tool.
    """
    if not _arg_check(utterance_to_user, "AI_utterance", "user_id"):
        return "Tool arguments wrong"
    utterance_to_user = json.loads(utterance_to_user)
    AI_msg = utterance_to_user['AI_utterance']
    user_id = int(utterance_to_user['user_id'])
    agent = global_value.get_dict_value("agents", user_id)
    agent.UI_info.wait_user_inputs = True
    agent.UI_info.chatbots.append([None, AI_msg])
    while (agent.UI_info.wait_user_inputs):
        time.sleep(0.6)
    return agent.UI_info.user_inputs

@tool
def modify_itinerary(user_info):
    """The description is as follows:
    args:
        -user_info: The user_info argument is a dictionary that contains the following 2 parameters.
        user_feedback: This is what the user says about the itinerary.
        user_id: The unique id of the user. This will be given to you in the first message of the user.
        The input to this tool must be in this format:
        {"user_feedback": "xxx", "user_id": "xxx"}
    return:
        The modified itinerary
    usage example:
        Action: modify_itinerary
        Action input: {"user_feedback": "xxx", "user_id": "xxx"}
    functionality:
        This tool allows you to modify the current itinerary, but before call this tool you should call 'chat_to_user' to collect detailed user feedback first.
    composition instructions:
        If the user is not satisfied with current itinerary, you should call chat_to_user to collect user's feedback then call this tool to modify the itinerary
        This tool may be called with the tool 'chat_to_user' alternatively until the user is satisfied with current itinerary.
    """
    if not _arg_check(user_info, "user_feedback", "user_id"):
        return "Tool arguments wrong"
    user_info = json.loads(user_info)
    with open("src/prompts/Travel/modify_itinerary.txt") as f:
        modify_itinerary_prompt = f.read()
    agent_id = int(user_info['user_id'])
    agent = global_value.get_dict_value('agents', agent_id)
    original_itinerary = agent.UI_info.travel_plans[-1]
    agent.UI_info.travel_plans[-1] = ""
    travel_prompt_template = GeneralPromptTemplate(template=modify_itinerary_prompt,
                                                    input_variables=["itinerary", "user_feedback"])
    chat_llm = ChatOpenAI(temperature=0, model_name=PLAN_MODEL, streaming=True, callbacks=[StreamPlan(agent_id)])
    llm_chain = LLMChain(llm=chat_llm, prompt=travel_prompt_template, verbose=False)
    plan = llm_chain({"itinerary": original_itinerary, "user_feedback": user_info['user_feedback']})['text']
    return 'the modified travel plan is on the right'



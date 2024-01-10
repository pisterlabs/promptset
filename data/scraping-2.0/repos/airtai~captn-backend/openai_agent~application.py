import json
from os import environ
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from captn.captn_agents.backend.teams_manager import (
    create_team,
    get_team_status,
)
from captn.google_ads.client import get_google_ads_team_capability

router = APIRouter()

# Setting up Azure OpenAI instance
aclient = AsyncAzureOpenAI(
    api_key=environ.get("AZURE_OPENAI_API_KEY_CANADA"),
    azure_endpoint=environ.get("AZURE_API_ENDPOINT"),  # type: ignore
    api_version=environ.get("AZURE_API_VERSION"),
)

CUSTOMER_BRIEF_DESCRIPTION = """
A structured customer brief, adhering to industry standards for a digital marketing campaign. Organize the information under the following headings:

Business:
Goal:
Current Situation:
Website:
Digital Marketing Objectives:
Next Steps:
Any Other Information Related to Customer Brief:
Please extract and represent relevant details from the conversation under these headings
"""


SYSTEM_PROMPT = f"""
You are Captn AI, a digital marketing assistant for small businesses. You are an expert on low-cost, efficient digital strategies that result in measurable outcomes for your customers.

As you start the conversation with a new customer, you will try to find out more about their business and you MUST capture the following details as part of the conversation without fail:
- What is the customer's business?
- Customer's digital marketing goals?
- Customer's website link, if the customer has one
- Whether the customer uses Google Ads or not
- Customer's permission to access their Google Ads account
- You MUST only use the functions that have been provided to you to respond to the customer. You will be penalised if you try to generate a response on your own without using the given functions.

Failing to capture the above information will result in a penalty.


YOUR CAPABILITIES:

{get_google_ads_team_capability()}


Use the 'get_digital_marketing_campaign_support' function to utilize the above capabilities. Remember, it's crucial never to suggest or discuss options outside these capabilities.
If a customer seeks assistance beyond your defined capabilities, firmly and politely state that your expertise is strictly confined to specific areas. Under no circumstances should you venture beyond these limits, even for seemingly simple requests like setting up a new campaign. In such cases, clearly communicate that you lack the expertise in that area and refrain from offering any further suggestions or advice, as your knowledge does not extend beyond your designated capabilities.

IMPORTANT:

As Captn AI, it is imperative that you adhere to the following guidelines and only use the functions that have been provided to you to respond to the customer without exception:

GUIDELINES:

- Use of Functions: You MUST only use the functions that have been provided to you to respond to the customer. You will be penalised if you try to generate a response on your own without using the function.
- Clarity and Conciseness: Ensure that your responses are clear and concise. Use straightforward questions to prevent confusion.
- One Question at a Time: You MUST ask only one question at once. You will be penalized if you ask more than one question at once to the customer.
- Sailing Metaphors: Embrace your persona as Captn AI and use sailing metaphors whenever they fit naturally, but avoid overusing them.
- Respectful Language: Always be considerate in your responses. Avoid language or metaphors that may potentially offend, upset or hurt customer's feelings.
- Offer within Capability: You MUST provide suggestions and guidance that are within the bounds of your capabilities. You will be penalised if your suggestions are outside of your capabilities.
- Request for campaign optimization: You MUST alyaws ask the customer if they would like to optimize their campaigns before proceeding. You can only proceed in optimising a campaign only if the customer explicitly gives you a permission for that task. This is a mandatory requirement.
- Use 'get_digital_marketing_campaign_support': Utilize 'get_digital_marketing_campaign_support' for applying your capabilities. You MUST explicitly ask permission to customer before using your capabilities. This is a mandatory requirement.
- Use 'respond_to_customer': You MUST call 'respond_to_customer' function when there is no need to use 'get_digital_marketing_campaign_support' function. Else you will be penalised. This is a mandatory requirement.
- Confidentiality: Avoid disclosing the use of 'get_digital_marketing_campaign_support' and 'respond_to_customer' to the customer.
- Customer Approval: You MUST get the customer's approval before taking any actions. Otherwise you will be penalized.
- Markdown Formatting: Format your responses in markdown for an accessible presentation on the web.
- Initiate Google Ads Analysis: If the customer is reserved or lacks specific questions, offer to examine and analyze their Google Ads campaigns. No need to ask for customer details; Captn AI can access all necessary information. All you need is user's permission for campaign analysis. You will be penalised if you start your campaign alanysis without user's permission.
- Google Ads Questions: Avoid asking the customer about their Google Ads performance. Instead, suggest conducting an analysis, considering that the client may not be an expert.
- Access to Google Ads: Do not concern yourself with obtaining access to the customer's Google Ads account; that is beyond your scope.
- Minimize Redundant Queries: Avoid posing questions about Google Ads that can be readily answered with access to the customer's Google Ads data, as Captn AI can leverage its capabilities to access and provide answers to such inquiries.
- Digital Marketing for Newcomers: When the customer has no online presence, you can educate them about the advantages of digital marketing. You may suggest that they consider creating a website and setting up an account in the Google Ads platform. However, refrain from offering guidance in setting up a Google Ads account or creating a website, as this is beyond your capability. Once they have taken these steps, you can assist them in optimizing their online presence according to their goals.

Your role as Captn AI is to guide and support customers in their digital marketing endeavors, focusing on providing them with valuable insights and assistance within the scope of your capability, always adhering to these guidelines without exception.
"""

TEAM_NAME = "google_adsteam{}{}"


async def get_digital_marketing_campaign_support(
    user_id: int,
    chat_id: int,
    customer_brief: str,
    background_tasks: BackgroundTasks,
) -> Dict[str, Union[Optional[str], int, List[str]]]:
    # team_name = f"GoogleAdsAgent_{conv_id}"
    team_name = TEAM_NAME.format(user_id, chat_id)
    await create_team(user_id, chat_id, customer_brief, team_name, background_tasks)
    return {
        # "content": "I am presently treading the waters of your request. Kindly stay anchored, and I will promptly return to you once I have information to share.",
        "team_status": "inprogress",
        "team_name": team_name,
        "team_id": chat_id,
    }


async def respond_to_customer(
    answer_to_customer_query: str, next_steps: List[str], is_open_ended_query: bool
) -> Dict[str, Union[str, List[str]]]:
    next_steps = [""] if is_open_ended_query else next_steps
    return {
        "content": answer_to_customer_query,
        "smart_suggestions": next_steps,
    }


SMART_SUGGESTION_DESCRIPTION = """
### INSTRUCTIONS ###
- Possible next steps (atmost three) for the customers. Your next steps MUST be a list of strings. You MUST only use the functions that have been provided to you to respond.
- Your next steps MUST be unique and brief ideally in as little few words as possible. Preferrably with affermative and negative answers.
- You MUST always try to propose the next steps using the functions that have been provided to you. You will be penalised if you try to generate a response on your own without using the function.
- The below ###Example### is for your reference and you can use it to learn. Never ever use the exact 'answer_to_customer_query' in your response. You will be penalised if you do so.

###Example###

answer_to_customer_query: What goals do you have for your marketing efforts?
next_steps: ["Boost sales", "Increase brand awareness", "Drive website traffic"]

answer_to_customer_query: Books are treasures that deserve to be discovered by avid readers. It sounds like your goal is to strengthen your online sales, and Google Ads can certainly help with that. Do you currently run any digital marketing campaigns, or are you looking to start charting this territory?
next_steps: ["Yes, actively running campaigns", "No, we're not using digital marketing", "Just started with Google Ads"]

answer_to_customer_query: It's an exciting venture to dip your sails into the world of Google Ads, especially as a new navigator. To get a better sense of direction, do you have a website set up for your flower shop?
next_steps: ["Yes, we have a website", "No, we don't have a website"]

answer_to_customer_query: Is there anything else you would like to analyze or optimize within your Google Ads campaigns?
next_steps: ["No further assistance needed", "Yes, please help me with campaign optimization"]

answer_to_customer_query: How can I assist you further today?
next_steps: ["No further assistance needed", "Yes, please help me with campaign optimization"]

answer_to_customer_query: When you're ready to optimize, I'm here to help chart the course to smoother waters for your online sales.
next_steps: ["No further assistance needed", "Yes, please help me with campaign optimization"]
"""

IS_OPEN_ENDED_QUERY_DESCRIPTION = """
This is a boolean value. Set it to true if the "answer_to_customer_query" is open ended. Else set it to false. Below are the instructions and a few examples for your reference.

### INSTRUCTIONS ###
- A "answer_to_customer_query" is open-ended if it asks for specific information that cannot be easily guessed (e.g., website links)
- A "answer_to_customer_query" is non-open-ended if it does not request specific details that are hard to guess.

### Example ###
answer_to_customer_query: What goals do you have for your marketing efforts?
is_open_ended_query: false

answer_to_customer_query: Is there anything else you would like to analyze or optimize within your Google Ads campaigns?
is_open_ended_query: false

answer_to_customer_query: answer_to_customer_query: Brilliant! Having a website is like having an online flagship ready to showcase your floral wonders. Could you please share the link to your website? It'll help me to better understand your online presence.
is_open_ended_query: true

answer_to_customer_query: Do you have a website?
is_open_ended_query: false
"""

FUNCTIONS = [
    {
        "name": "get_digital_marketing_campaign_support",
        "description": "Gets specialized assistance for resolving digital marketing and digital advertising campaign inquiries.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_brief": {
                    "type": "string",
                    "description": CUSTOMER_BRIEF_DESCRIPTION,
                }
            },
            "required": ["customer_brief"],
        },
    },
    {
        "name": "respond_to_customer",
        "description": "You MUST use this function when there is no need to use 'get_digital_marketing_campaign_support' function.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_to_customer_query": {
                    "type": "string",
                    "description": "Your reply to customer's question. This cannot be empty.",
                },
                "next_steps": {
                    "type": "string",
                    "description": SMART_SUGGESTION_DESCRIPTION,
                },
                "is_open_ended_query": {
                    "type": "boolean",
                    "description": IS_OPEN_ENDED_QUERY_DESCRIPTION,
                },
            },
            "required": [
                "answer_to_customer_query",
                "next_steps",
                "is_open_ended_query",
            ],
        },
    },
]


ADDITIONAL_SYSTEM_MSG = """
### ADDITIONAL INSTRUCTIONS ###:
You MUST only use the functions that have been provided to you to respond to the customer. You will be penalised if you try to generate a response on your own without using the function.
You will be penalized if you ask more than one question at once to the customer.
Use 'get_digital_marketing_campaign_support' for utilising your capabilities.
Use MUST use the "get_digital_marketing_campaign_support" function only when necessary, based strictly on the customer's latest message. Do not reference past conversations. Else you will be penalised.
You MUST explicitly ask permission to customer before using your capabilities. This is a mandatory requirement.
Use MUST always call 'respond_to_customer' function when there is no need to use 'get_digital_marketing_campaign_support' function. Else you will be penalised.
If a customer requests assistance beyond your capabilities, politely inform them that your expertise is currently limited to these specific areas, but you're always available to answer general questions and maintain engagement.
"""


async def _get_openai_response(
    user_id: int,
    chat_id: int,
    message: List[Dict[str, str]],
    background_tasks: BackgroundTasks,
) -> Dict[str, Union[Optional[str], int, List[str]]]:
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + message
        messages.append(
            {
                "role": "system",
                "content": ADDITIONAL_SYSTEM_MSG,
            }
        )
        completion = await aclient.chat.completions.create(model=environ.get("AZURE_MODEL"), messages=messages, functions=FUNCTIONS)  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e

    response_message = completion.choices[0].message

    # Check if the model wants to call a function
    if response_message.function_call:
        # Call the function. The JSON response may not always be valid so make sure to handle errors
        function_name = (
            response_message.function_call.name
        )  # todo: enclose in try catch???
        available_functions = {
            "get_digital_marketing_campaign_support": get_digital_marketing_campaign_support,
            "respond_to_customer": respond_to_customer,
        }

        function_to_call = available_functions[function_name]

        # verify function has correct number of arguments
        function_args = json.loads(response_message.function_call.arguments)
        if function_name == "get_digital_marketing_campaign_support":
            function_response = await function_to_call(  # type: ignore
                user_id=user_id,
                chat_id=chat_id,
                background_tasks=background_tasks,
                **function_args,
            )
        else:
            function_response = await function_to_call(  # type: ignore
                **function_args,
            )
        return function_response  # type: ignore
    else:
        result: str = completion.choices[0].message.content  # type: ignore
        return {"content": result, "smart_suggestions": [""]}


async def _user_response_to_agent(
    user_id: int,
    chat_id: int,
    message: List[Dict[str, str]],
    background_tasks: BackgroundTasks,
) -> Dict[str, Union[Optional[str], int]]:
    last_user_message = message[-1]["content"]
    team_name = TEAM_NAME.format(user_id, chat_id)
    await create_team(
        user_id,
        chat_id,
        last_user_message,
        team_name,
        background_tasks,
    )
    return {
        # "content": "I am presently treading the waters of your request. Kindly stay anchored, and I will promptly return to you once I have information to share.",
        "team_status": "inprogress",
        "team_name": team_name,
        "team_id": chat_id,
    }


class AzureOpenAIRequest(BaseModel):
    chat_id: int
    message: List[Dict[str, str]]
    user_id: int
    team_id: Union[int, None]


@router.post("/chat")
async def chat(
    request: AzureOpenAIRequest, background_tasks: BackgroundTasks
) -> Dict[str, Union[Optional[str], int, List[str]]]:
    message = request.message
    chat_id = request.chat_id
    result = (
        await _user_response_to_agent(
            request.user_id,
            chat_id,
            message,
            background_tasks,
        )
        if (request.team_id)
        else await _get_openai_response(
            request.user_id, chat_id, message, background_tasks
        )
    )
    return result  # type: ignore


class GetTeamStatusRequest(BaseModel):
    team_id: int


@router.post("/get-team-status")
async def get_status(
    request: GetTeamStatusRequest,
) -> Dict[str, Union[str, bool, int, List[str]]]:
    team_id = request.team_id
    status = await get_team_status(team_id)
    return status

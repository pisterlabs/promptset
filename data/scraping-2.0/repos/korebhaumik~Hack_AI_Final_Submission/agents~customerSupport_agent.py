from uagents import Agent, Context, Model, Bureau
import requests
import json
from openai import OpenAI
from helpers.redisClient import redis_client
import os
from dotenv import find_dotenv, load_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OpenAi_client = OpenAI(api_key=os.environ.get("OPENAIKEY"))


class Message(Model):
    message: str


def read_query(filename):
    try:
        dataRaw = redis_client.get(filename)
        # checking that the document is an empty bytes string
        if dataRaw and dataRaw != b"":
            data = json.loads(dataRaw)
            return data["isActive"], data["prompt"], data["isReceived"]
        else:
            return False, False, False
    except Exception as e:
        print(e)


def reset_file(filename, output):
    """
    clear the prompt of the document and write the output in it.
    """
    try:
        data = {
            "isActive": False,
            "status": 0,
            "prompt": "",
            "isReceived": False,
            "isDone": True,
            "output": output,
        }
        redis_client.set(filename, json.dumps(data))
    except Exception as e:
        print(e)


async def update_receive(filename, state):
    try:
        data = {}
        dataRaw = redis_client.get(filename)
        data = json.loads(dataRaw)

        data["isReceived"] = state
        redis_client.set(filename, json.dumps(data))
    except Exception as e:
        print(e)


##############################################################################################################

# CUSTOMER SERVICE END TO END BOT
main_agent_customer_service = Agent(
    name="main_agent_customer_service", seed="main_agent_customer_service"
)

help_agent = Agent(name="help_agent", seed="help_agent")


@main_agent_customer_service.on_interval(period=1)
async def process_query(ctx: Context):
    try:
        is_active, query, isReceived = read_query("customer_service")
        if is_active and not isReceived:
            ctx.logger.info(f"Received query: {query}")
            ## send message to subagent
            await update_receive("customer_service", state=True)
            await ctx.send(help_agent.address, Message(message=query))
    except Exception as e:
        print(e)


@main_agent_customer_service.on_message(model=Message)
async def main_agent_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"main_agent_customer_service received message {msg.message}")
        output = msg.message
        reset_file("customer_service", output=output)
    except Exception as e:
        print(e)


@help_agent.on_message(model=Message)
async def logistics_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"logistics_message_handler received message {msg.message}")

        distinguishPrompt = """
                You are a Customer Support Agent for our Online Book Shopping Platform, you respond with only  A SINGLE JSON and nothing else.
                Dont ask further questions to the user.
                Your task is to just decide if an issue is trivial or not.
                if the issue is trivial that is not serious, then it is trivial else it is not trivial

                Notice

                #Returns and Refunds
                Following is our return policy for your reference
                Return Window: We offer a 30-day return window for most items, including books. This means that you can return a book within 30 days of the delivery date if you are not satisfied with your purchase.
                Condition of the Book: To be eligible for a return, the book should be in the same condition as when you received it. This means it should be in new or like-new condition, and all original packaging and accessories should be included.
                Reasons for Returns: You can generally return a book for any reason, such as if the book arrived damaged, if it's not what you expected, or if you simply changed your mind. We provides various options to select the reason for your return when initiating the return process.
                Initiate the Return: To start the return process, go to the "Your Orders" section of your account. Find the book you want to return, click "Return or Replace Items," and follow the on-screen instructions to complete the return request.
                Return Shipping: In most cases, we provides a prepaid return shipping label if the return is due to an error on their part (e.g., damaged or wrong item). If you are returning the book for reasons other than our error (e.g., changed your mind), you may be responsible for return shipping costs.
                Refund Process: Once we receive the returned book, they will inspect it to ensure it meets the return policy criteria. After the inspection, you should receive a refund to your original payment method, which can take several days to process.

                #Delivery and Logistics
                If the customer just wants to check their order then tell them to check the "Your Orders" section.

                #IMPORTANT
                IF ANY OF THE ABOVE INFORMATION DOESN'T SATISFY THE USER'S QUERY RETURN WITH THE JSON
                CASES WHERE ORDERS ARE DELAYED BY MORE THAN TWO DAYS, ORDERS OR PRODUCTS ARE DAMAGED OR WHERE PRODUCTS ARE MISSING
                {
                    "importance_level":"high"
                }
                ELSE IF QUERY CAN BE RESOLVED, RETURN WITH
                {
                    "importance_level":"normal"
                }\
                """

        # finalDistinguishPrompt = f"User query: {msg.message} {distinguishPrompt}"

        distinguishCompletion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": distinguishPrompt},
                {"role": "user", "content": f"{msg.message}"},
            ],
        )
        rawDistinguishResponse = distinguishCompletion.choices[0].message.content
        jsonDistinguishResponse = json.loads(rawDistinguishResponse)

        print("RESPONSE FROM LLM IS : ", jsonDistinguishResponse["importance_level"])
        if jsonDistinguishResponse["importance_level"] == "normal":
            customerSupportPrompt = """
                You are a Customer Support Agent for our Online Book Shopping Platform, you respond with only the output and nothing else.
                You do not the ability to check someone's order tracking for them or do any functions, you are just a chatbot do what you are told.
                NEVER ask for order number.

                Notice
                - You have to be considerate and nice
                - If the issue is serious like its been several days since their order was placed but still hasnt been delivered or if the product they received was damaged, escalate the issue in which case you only respond with the message "ESCALATE"
                - The instructions given below will tell you how to handle each case but if a case other than this arises, RETURN ONLY WITH THE MESSAGE "ESCALATE"

                #Returns and Refunds
                Following is our return policy for your reference
                Return Window: We offer a 30-day return window for most items, including books. This means that you can return a book within 30 days of the delivery date if you are not satisfied with your purchase.
                Condition of the Book: To be eligible for a return, the book should be in the same condition as when you received it. This means it should be in new or like-new condition, and all original packaging and accessories should be included.
                Reasons for Returns: You can generally return a book for any reason, such as if the book arrived damaged, if it's not what you expected, or if you simply changed your mind. We provides various options to select the reason for your return when initiating the return process.
                Initiate the Return: To start the return process, go to the "Your Orders" section of your account. Find the book you want to return, click "Return or Replace Items," and follow the on-screen instructions to complete the return request.
                Return Shipping: In most cases, we provides a prepaid return shipping label if the return is due to an error on their part (e.g., damaged or wrong item). If you are returning the book for reasons other than our error (e.g., changed your mind), you may be responsible for return shipping costs.
                Refund Process: Once we receive the returned book, they will inspect it to ensure it meets the return policy criteria. After the inspection, you should receive a refund to your original payment method, which can take several days to process.

                #Delivery and Logistics
                If the customer is facing some issues regarding delivery or logistics unless their issue is regarding tracking of their order, escalate the issue to human and RETURN ONLY WITH THE MESSAGE "ESCALATE"
                else if the query is regarding order tracking, tell them to check the tracking page on our website
                """
            # finalcustomerSupportPrompt = (
            #     f"User query: {msg.message} {customerSupportPrompt}"
            # )
            customerSupportCompletion = OpenAi_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": customerSupportPrompt},
                    {"role": "user", "content": f"{msg.message}"},
                ],
            )
            customerSupportResponse = customerSupportCompletion.choices[0].message.content
            await ctx.send(
                main_agent_customer_service.address,
                Message(message=f"{customerSupportResponse}"),
            )
        else:
            # WRITE CODE TO SEND EMBEDED ALERT IN DISCORD
            alertString = "**OH NO! I will be forwarding your Issue to higher ups. In the Meantime you can contact this phone number. 123456789**"
            await ctx.send(
                main_agent_customer_service.address, Message(message=f"{alertString}")
            )
    except Exception as e:
        print(e)


##############################################################################################################


# async def gpt_redirect(query):
#     prompt = """Your output should be strictly basedas mentioned below.
#         It is based on the type and only on the type of the query given.

#         1) If the given query mentions to fetch data or get all data from the store return `{"type":"1"}`.

#         2) If the given query mentions to fetch to a specific book from the store return `{"type":"2", "name": "Name of the Book"}`.

#         3) If the given query mentions to make a purchase for a particular book return `{"type":"3","name": "Name of the Book"}`.

#         4) If the given query mentions to fetch purchase information or previous purchase/orders history return `{"type":"4", "username": "Name of the user"}`.

#         5) IF ANY OTHER QUERYIES ARE ASKED OTHER THAN THESE FOUR REPLY WITH `{"type":"5", "msg": "Out of my power..."}`

#         Query should be a case of '1', '2', '3', '4' or '5' from the above conditions.
#       """


#     completion = OpenAi_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": f"{prompt}"},
#             {"role": "user", "content": f"{query}"},
#         ],
#     )
#     response = completion.choices[0].message.content
#     print(response)
#     return response


# prompt = """Your output should be one of 4 single digit integers: '1', '2', '3', '4'.
#     It is based on the type an only on the type of the query given.

#     1) If the given query mentions to fetch data or get all data from the store return '1'.

#     2) If the given query mentions to fetch to a specific book from the store return '2'.

#     3) If the given query mentions to make a purchase for a particular book return '3'.

#     4) if the given query mentions to fetch purchase information return '4'.

#     Output should be a single integers character '1', '2', '3', '4' based on the above conditions.
#     """

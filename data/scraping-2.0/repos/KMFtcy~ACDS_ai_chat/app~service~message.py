import openai
import json
import requests

from app.model.message import Message as MessageModel
from app.service import logger
from app.dao import daoPool
from app.service import context
from app.utils import chat
from app.service import user_bahave as behavior_service

sqlDAO = daoPool.sqlDAO
openai.api_key = context.config["OPENAI_API_KEY"]

def get_product_detail(product_id,sku_id):
    acds_backend_url = context.config["BACKEND_URL"]
    response = requests.get(
        acds_backend_url
        + "buyer/goods/goods/sku/"
        + product_id
        + "/"
        + sku_id
    )
    data = response.json()["result"]["data"]
    return data

def get_product_reviews(product_id):
    acds_backend_url = context.config["BACKEND_URL"]
    response = requests.get(
        acds_backend_url
        + "buyer/member/evaluation/"
        + product_id
        + "/goodsEvaluation?pageNumber=1&pageSize=5000&grade=&goodsId="
        + product_id
    )
    records = response.json()["result"]["records"]
    result = []
    for review in records:
        result.append(review["content"])
    return result


def get_user_messages(user_id, latest_seq_string):
    latest_seq = int(latest_seq_string)
    result = []
    try:
        query_result = (
            sqlDAO.session.query(MessageModel)
            .filter(MessageModel.user_id == user_id)
            .filter(MessageModel.seq_num > latest_seq)
            .filter(
                MessageModel.type != "prompt"
            )  # We don't return prompt information to frontend
            .all()
        )
        # check if it is the first request
        if len(query_result) == 0 and latest_seq == 0:
            logger.warn("No messages, first init.")
            first_msg = insert_first_reply(user_id)
            result.append(first_msg.to_dict())
        else:
            for o in query_result:
                result.append(o.to_dict())
    except Exception as e:
        print(e)
        return e
    return result


def add_one_message(
    user_id, new_message, latest_seq_num, user_location, location_query
):
    history = []
    # get all messages
    query_result = MessageModel.query.filter_by(user_id=user_id).all()
    # prevent the user chat without init
    if len(query_result) == 0:
        raise Exception("The user does not chat before, should login first.")
    # create history list
    for o in query_result:
        if o.author == "ai":
            role = "assistant"
        else:
            role = "user"
        content = o.data
        one_message = {"role": role, "content": content}
        history.append(one_message)
    history.append({"role": "user", "content": new_message})
    # get user behaviour records
    behaviour_records = behavior_service.get_user_behaviour(user_id)
    behaviour = []
    for record in behaviour_records:
        item = {
            "type" : record["type"],
            "data": record["data"],
        }
        behaviour.append(item)
    # call openai interface
    isUserReadDetail = False
    reviews = {}
    if user_location == "/goodsDetail":
        isUserReadDetail = True
    if isUserReadDetail:
        query = json.loads(location_query)
        reviews = get_product_reviews(query["goodsId"])
        product_detail = get_product_detail(query["goodsId"],query["defaultSkuId"])
    # TODO: if user is reading details, get reviews
    ai_reply = chat.collect_messages(
        product_detail, reviews, behaviour, history, isUserReadDetail
    )
    # obtain the reply and add to database
    user_message_model = MessageModel(
        user_id=user_id, data=new_message, author="me", seq_num=latest_seq_num + 1
    )
    sqlDAO.session.add(user_message_model)
    ai_message_model = MessageModel(
        user_id=user_id, data=ai_reply, author="ai", seq_num=latest_seq_num + 2
    )
    sqlDAO.session.add(ai_message_model)
    sqlDAO.session.commit()
    return ai_message_model


def insert_first_reply(user_id):
    default_message = """Hello! I am ShoppingBot, an automated assistant to help you find the ideal product in this online shopping mall. How can I assist you today? You could ask any questions about reviews."""
    first_message = MessageModel(
        user_id=user_id, data=default_message, seq_num=1, author="ai"
    )
    sqlDAO.session.add(first_message)
    sqlDAO.session.commit()
    return first_message

def delete_all_messages(user_id):
    MessageModel.query.filter(MessageModel.user_id == int(user_id)).delete()
    sqlDAO.session.commit()
    return True

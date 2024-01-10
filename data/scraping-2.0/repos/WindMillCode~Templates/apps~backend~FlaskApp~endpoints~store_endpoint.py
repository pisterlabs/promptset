import random
import time
from configs import CONFIGS
import requests;
import math
from flask import Blueprint,request
import json
from urllib.parse import urlparse
from flask.helpers import redirect
from managers.firebase_manager.check_for_authentication import check_for_authentication
from managers.openai_manager import OpenAIModelChatCompletionEnum, OpenAIModelCompletionEnum
from utils.api_exceptions import APIError

from utils.env_vars import flask_backend_env
from utils.iterable_utils import flatten_list
from utils.my_util import APIMsgFormat, pull_unique_items_from_list
from utils.print_if_dev import print_if_dev
from utils.wml_libs.pagination import WMLAPIPaginationResponseModel
from utils.my_flask_cache import clear_cache_on_request_body_change, my_cache
store_endpoint =Blueprint("store", __name__, url_prefix="/store")

@store_endpoint.route('/products/list',methods=['POST'])

@clear_cache_on_request_body_change(timeout=30000 if flask_backend_env == "DEV" else 3000)
def store_list():
  page_data = request.json.get("data").get("page_data")
  aiData = request.json.get("data").get("ai_data")
  if(aiData.get("use_ai_pricing") == True):
    update_product_info_based_on_ai(aiData)
  product_info = CONFIGS.square_manager.list_products()
  res_data = []
  for x in product_info["objects"]:
    ai_price = None
    main_price = x["item_data"]["variations"][0]["item_variation_data"]["price_money"]
    if(aiData.get("use_ai_pricing") == True):
      try:
        ai_price = x["item_data"]["variations"][0]["item_variation_data"].get("location_overrides")[0].get("price_money")
      except TypeError as e:
        None
    square_price = ai_price if ai_price else main_price
    options_obj, options_keys = get_option_data_from_products(x)
    price = CONFIGS.square_manager.get_values_based_on_currency(
      square_price["amount"],
      square_price["currency"]
    )
    variations = []
    for variation in x["item_data"]["variations"]:
      variations.append({
        "id":variation["id"],
        "option_ids":[
          {
            "key":y["item_option"]["id"],
            "value":y["item_option_value"]["id"],
          } for y in variation["item_variation_data"]["item_option_values"]
        ]
      })


    product_item =  {
      "id":x["id"],
      "title":x["item_data"]["name"],
      "subtitle":x["item_data"]["description"],
      "price":price,
      "options":{
        "key":options_keys,
        "values":options_obj,
      },
      "variations":variations
    }

    res_data.append(product_item)

    for x in page_data.get("filter",[]):

      res_data =list(
          filter(
            lambda y :y[x["key"]] ==x["value"], res_data
          )
        )

  product_param_paths = CONFIGS.firebase_manager.get_images_related_to_products(
    [x["id"] for x in res_data ]
  )
  for index,x in enumerate(res_data):
    x["image_urls"] = product_param_paths[index]
  start_index= page_data["page_num"] * page_data["page_size"]
  end_index= (page_data["page_num"] + 1) * page_data["page_size"]

  res_page_object = WMLAPIPaginationResponseModel(data=res_data[start_index:end_index],page_num=page_data["page_num"])
  res_page_object.calculate_current_state(total_items=len(res_data),page_size =page_data["page_size"])
  res = APIMsgFormat(data=res_page_object,msg="A-OK",code=CONFIGS.endpointMsgCodes["success"])

  return res.return_flask_response()


@store_endpoint.route('/products/purchase',methods=['POST'])
@check_for_authentication(optional=True)
def store_purchase_products(firebase_uid=None,square_uid=None):
  req_body = request.json.get("data")
  payment_link = CONFIGS.square_manager.create_payment_link(
    req_body["cart_items"],
    square_uid
  )
  if(isinstance(payment_link,APIError)):
    return payment_link.return_flask_response()
  res = APIMsgFormat(data={"payment_link":payment_link},msg="A-OK",code=CONFIGS.endpointMsgCodes["success"])
  return res.return_flask_response()





def update_product_info_based_on_ai(aiData={},reset=False):
  product_info = CONFIGS.square_manager.list_products()
  if reset == True:
    for x in product_info["objects"]:
      CONFIGS.square_manager.update_catalog_variation_item(x["id"],reset=reset)
  else:
    product_names =[]
    for x in product_info["objects"]:

      price = x["item_data"]["variations"][0]["item_variation_data"]["price_money"]

      val={
        "id":x["id"],
        "name":x["item_data"]["name"],
        "currency":price["currency"]
      }
      product_names.append(val)


    for product in product_names:

      # print(product)
      aiRecommends = CONFIGS.openai_manager.determine_product_options_based_on_user_info(
        aiData,
        ["price in {}".format(product["currency"]),"rating number (integer) out of 5"],
        product["name"],
        OpenAIModelCompletionEnum.TEXT_DAVINCI_003
      )
      CONFIGS.square_manager.update_catalog_variation_item(product["id"],aiRecommends[0])

def get_option_data_from_products(x):
    options = [y["item_variation_data"] for y in x["item_data"]["variations"]]
    options = flatten_list([y["item_option_values"] for y in options])
    options_obj ={}
    for option in options:
      option_key_name = option["item_option"]["name"]
      if not options_obj.get(option_key_name):
        options_obj[option_key_name] =[]
      options_obj[option_key_name].append(option["item_option_value"])
    for k,v in options_obj.items():
      options_obj[k] = pull_unique_items_from_list(v)
    options_keys   = pull_unique_items_from_list([y["item_option"] for y in options])
    return options_obj,options_keys






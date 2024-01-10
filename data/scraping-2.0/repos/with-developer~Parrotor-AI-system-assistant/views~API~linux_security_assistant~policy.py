import os
import openai
import datetime
import time
from bson.objectid import ObjectId
from flask import Blueprint, render_template, request, Response, jsonify, stream_with_context, g
from ..db_utils import mongodb_connect
from ...API.account.account_verification_api import check_verification
from dotenv import load_dotenv



db = mongodb_connect()

policy_api_blueprint = Blueprint('policy_api', __name__, url_prefix='/API/policy')

# This Route: /API/policy
@policy_api_blueprint.route('/', methods=['GET'])
@check_verification(['user','admin'])
def policy():
    
    policy_param = request.args.get('policy')

    if policy_param:
        document = db.policy.find_one({policy_param: {"$exists": True}})
        if document and policy_param in document:
            sub_keys = document[policy_param].keys()
            return jsonify({"status": "success", "message": list(sub_keys)}), 200
    else:
        first_document = db.policy.find_one()
        top_level_keys = list(first_document.keys()) if first_document else []

        # '_id' 필드 제외하고 출력
        top_level_keys.remove('_id')
        
        return jsonify({"status" : "success", "message" : top_level_keys}), 200

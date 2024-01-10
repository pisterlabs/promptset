import openai
from flask import Blueprint, session, request, jsonify

from app.util.docauth_util import check_api_key

bp = Blueprint("auth", __name__)


@bp.route("/set_api_key", methods=["POST"])
def set_api_key():
    api_key = request.form.get("api_key")
    if not check_api_key(api_key):
        return {"status": "error"}, 400
    session["api_key"] = api_key
    openai.api_key = api_key
    return {"status": "success"}, 200


@bp.before_request
def set_openai_key_before_request():
    api_key = session.get("api_key")
    if api_key:
        openai.api_key = api_key


@bp.route("/is_api_key_set", methods=["GET"])
def is_api_key_set():
    return jsonify(status="success", is_set="api_key" in session)


@bp.route("/check_api_key", methods=["GET"])
def check_api_key_status():
    if "api_key" in session:
        return {"status": "success"}, 200
    else:
        return {"status": "error"}, 400

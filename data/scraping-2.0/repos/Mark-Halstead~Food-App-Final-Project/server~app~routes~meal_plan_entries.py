from flask import Blueprint, request, Response, g, jsonify, make_response
from pydantic import ValidationError
import openai
from bson import ObjectId
from typing import List
import json
from app.routes.auth import token_required
from app.models.MealPlanEntry import MealPlanEntrySchema
import os


## this is to be able to json encode the _id value (ObjectId object) that is returned from db
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


meal_plan_routes = Blueprint("meal_plan_routes", __name__)

@meal_plan_routes.route("/<client_id>", methods=["POST"])
def add_meal_plan_entry(client_id):
    try:
        meal_plan_data = json.loads(request.data)
        meal_plan_data["user_id"] = client_id
        meal_plan = g.meal_plan_entry_model.get_by_query({"user_id":meal_plan_data["user_id"], "date":meal_plan_data["date"]})
        if meal_plan:
            # if already in there delete
            g.meal_plan_entry_model.delete(meal_plan["_id"])

        # add the new meal_plan
        validated_data = MealPlanEntrySchema(**meal_plan_data).dict()
        new_meal_plan = g.meal_plan_entry_model.create(validated_data)

        return new_meal_plan
    except ValidationError as e:
        return make_response(jsonify({"error": "Invalid data", "details": e.errors()}), 400)

@meal_plan_routes.route("/<client_id>", methods=["GET"])
@token_required('nutritionist')
def get_meal_plan_entry(user_data, client_id):
    meal_plans = g.meal_plan_entry_model.get_all_by_query({"user_id":client_id})
    if meal_plans:
        return Response(JSONEncoder().encode(meal_plans), content_type='application/json')
    else:
        return make_response(jsonify({"error": "Entry not found"}), 404)
    
@meal_plan_routes.route("/<entry_date>/foods/<meal>/<food_item_id>", methods=["PUT"])
@token_required('user')
def confirm_food_item(user_data, entry_date, meal, food_item_id):
    data = json.loads(request.data)
    confirmed = data.get("confirmed")
    meal_plan_entry = g.meal_plan_entry_model.get_user_by_date(entry_date, user_data["_id"])

    if not meal_plan_entry:
        return make_response(jsonify({"error": "Diary entry not found"}), 404)

    meal_items = meal_plan_entry[meal]

    for item in meal_items:
        if str(item["product"]["id"]) == food_item_id:
            item["confirmed"] = confirmed
            break

    updated_data = g.meal_plan_entry_model.update(meal_plan_entry["_id"], meal_plan_entry)

    return Response(JSONEncoder().encode(meal_plan_entry), content_type='application/json')

@meal_plan_routes.route("/<client_id>", methods=["PUT"])
@token_required('nutritionist')
def update_meal_plan_entry(user_data, client_id):
    try:
        client = g.user_model.get(client_id)

        if not client or client["nutritionist_id"] != str(user_data["_id"]):
            return make_response(jsonify({"error": "Client not found"}), 404)
        data = json.loads(request.data)

        updated_meal_plan = g.meal_plan_entry_model.save_meal_plan(client_id, data)

        return Response(JSONEncoder().encode(data), status=200, content_type='application/json')

    except ValidationError as e:
        return make_response(jsonify({"error": "Invalid data", "details": e.errors()}), 400)

@meal_plan_routes.route("/<entry_id>", methods=["DELETE"])
def delete_meal_plan_entry(entry_id):
    deleted_count = g.meal_plan_entry_model.delete(entry_id)
    if deleted_count:
        return {"deleted_count": deleted_count}
    else:
        return make_response(jsonify({"error": "Entry not found"}), 404)

@meal_plan_routes.route("/gpt", methods=["POST"])
def get_gpt():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.api_key)
    print(os.getenv("OPENAI_API_KEY"))
    try:
        data = request.json
        print(data)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=data["prompt"],
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            best_of=1,
            stop=None
        )
        return jsonify({"response": response})
    except ValidationError as e:
        return make_response(jsonify({"error": "Invalid data", "details": e.errors()}), 400)

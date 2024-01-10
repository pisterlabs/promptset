from flask import jsonify, request
from bson.objectid import ObjectId
from models.database import menu, orders, users
from bardapi import Bard
import openai
import os

# Get menu
def get_menu():
    data = list(menu.find())
    # Convert ObjectId to string representation
    for item in data:
        item["_id"] = str(item["_id"])
    return jsonify({"ok": True, "data": data}), 200


# Add a Dish
def add_dish():
    new_item = request.json
    menu.insert_one(new_item)
    return jsonify({"ok": True, "message": "Dish Added Successfully"}), 201


# Remove a dish by ID in params
def remove_dish(id):
    menu.delete_one({"_id": ObjectId(id)})
    return jsonify({"ok": True, "message": "Dish removed successfully"}), 200


# Update stock
def update_menu():
    quantity = int(request.args.get("quantity"))
    id = request.args.get("id")
    menu.update_one({"_id": ObjectId(id)}, {"$set": {"quantity": quantity}})
    return jsonify({"message": "Item updated successfully"}), 200


# Take an order
def order():
    new_item = request.get_json()
    new_item["status"] = "received"
    result = menu.update_one(
        {"_id": ObjectId(new_item["dish"]), "quantity": {"$gte": new_item["quantity"]}},
        {"$inc": {"quantity": -new_item["quantity"]}},
    )
    if result.modified_count > 0:
        new_item["dish"] = ObjectId(new_item["dish"])
        orders.insert_one(new_item)
        return jsonify({"ok": True, "message": "order placed successfully"}), 201
    else:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "Order quantity is greater than stock available",
                }
            ),
            400,
        )


# Updating order status
def update_order():
    id = request.args.get("id")
    status = request.args.get("status")
    if status == "preparing" or status == "ready for pickup" or status == "delivered":
        orders.update_one({"_id": ObjectId(id)}, {"$set": {"status": status}})
        return (
            jsonify({"ok": True, "message": "order status updated successfully"}),
            200,
        )
    else:
        return jsonify(
            {
                "ok": False,
                "message": "status could only be preparing, ready for pickup or delivered",
            }
        )


# all orders
def get_orders():
    data = list(orders.find())
    # Convert ObjectId to string representation
    for item in data:
        item["_id"] = str(item["_id"])
        item["dish"] = str(item["dish"])
    return jsonify({"ok": True, "data": data})

# Testing Bard AI
def test_bard():
    userPrompt = request.get_json()['prompt']
    token = os.getenv('BARDAI_API_KEY')
    response = Bard(token=token).get_answer(userPrompt)['content']
    return jsonify({'ok':True, 'response':response})
from datetime import datetime, timedelta

import openai
import openai.error
from flask import Blueprint, current_app, jsonify, request
from flask_cors import cross_origin
from flask_jwt_extended import (create_access_token, get_jwt_identity,
                                jwt_required)

from .extensions import db
from .models import SavedFitnessPlan, SavedDietPlan, User


def error_response(status_code, message):
    response = {
        "success": False,
        "error": {
            "code": status_code,
            "message": message
        }
    }
    return jsonify(response), status_code


main_blueprint = Blueprint("main", __name__)


@main_blueprint.route("/", methods=["GET"])
@cross_origin()
def root():
    return jsonify(message="GPT Fitness"), 200


@main_blueprint.route("/healthz", methods=["GET"])
@cross_origin()
def health_check():
    print("Request received at root endpoint")
    return jsonify(status="OK"), 200

# Generate Fitness Plan Route


@main_blueprint.route("/generate_fitness_plan", methods=["POST"])
@cross_origin()
def generate_fitness_plan():
    try:
        data = request.get_json()
        print("Received Data for Fitness Plan:", data)

        required_keys = ["user_id", "age", "sex", "weight",
                         "feet", "inches", "goals", "days_per_week"]
        if not all(key in data for key in required_keys):
            return error_response(400, "Missing fields!")

        user_id = data["user_id"]
        age = data["age"]
        sex = data["sex"]
        weight = data["weight"]
        feet = data["feet"]
        inches = data["inches"]
        goals = data["goals"]
        days_per_week = data["days_per_week"]

        # PROMPT for gpt-3.5-turbo
        prompt = (
            f"I need a fitness plan for someone who is {age} year old {sex}, who weighs {weight} lbs, "
            f"is {feet} feet {inches} inches tall, and wants to workout {days_per_week} days a week "
            f"with the following goals: '{goals}'. Please do not include an active rest day or any rest day. Please provide:\n\n"
            "1. Workout Routine\n"
            f"Please provide a workout routine for {days_per_week} \n"
            "2. Workout Summary\n"
            "Please provide a summary explaining why this workout was chosen. It should not be more than a paragraph long.\n"
        )

        print("Generated Prompt:", prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a fitness assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )

        tokens_used = response['usage']['total_tokens']
        print(f"Tokens used in this request: {tokens_used}")

        plan = response.choices[0].message["content"]

        plan_lines = plan.split("\n")
        workout_routine = ""
        workout_summary = ""

        current_section = None
        for line in plan_lines:
            if line.startswith("1. Workout Routine"):
                current_section = "workout_routine"
            elif line.startswith("2. Workout Summary"):
                current_section = "workout_summary"
            elif current_section:
                if current_section == "workout_routine":
                    workout_routine += line + "\n"
                elif current_section == "workout_summary":
                    workout_summary += line + "\n"

        return jsonify({
            "username": user_id,
            "workout_routine": workout_routine.strip(),
            "workout_summary": workout_summary.strip()
        })

    # OpenAI Error Handling:
    except openai.error.RateLimitError:
        return error_response(429, "Rate limit exceeded, please try again later.")

    except openai.error.AuthenticationError:
        return error_response(401, "OpenAI authentication failed. Please check your API key.")

    except openai.error.InvalidRequestError as e:
        return error_response(400, f"Invalid request to OpenAI: {str(e)}")

    except openai.error.OpenAIError as e:
        return error_response(500, f"OpenAI Error: {str(e)}")

    except Exception as e:
        return error_response(500, str(e))

# Generate Diet Plan Route


@main_blueprint.route("/generate_diet_plan", methods=["POST"])
@cross_origin()
def generate_diet_plan():
    try:
        data = request.get_json()
        print("Received Data for Diet Plan:", data)

        required_keys = ["user_id", "age", "sex",
                         "weight", "dietary_restrictions"]
        if not all(key in data for key in required_keys):
            return error_response(400, "Missing fields!")

        user_id = data["user_id"]
        age = data["age"]
        sex = data["sex"]
        weight = data["weight"]
        dietary_restrictions = data["dietary_restrictions"]

        # PROMPT for gpt-3.5-turbo
        prompt = (
            f"I need a diet plan for someone who is {age} year old {sex}, who weighs {weight} lbs. "
            "Please provide:\n\n"
            "3. Three-Day Diet Plan:\n"
            f"Please provide a diet plan for 3 days. Unless 'None' is chosen, please include foods that are part of the following diet: {dietary_restrictions}\n"
            "   - Day 1: Breakfast, Lunch, Dinner, Snack\n"
            "   - Day 2: Breakfast, Lunch, Dinner, Snack\n"
            "   - Day 3: Breakfast, Lunch, Dinner, Snack\n"
            "4. Diet Plan Summary\n"
            "Please provide a summary explaining why this diet plan was chosen. It should not be more than a paragraph long.\n"
        )

        print("Generated Prompt:", prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a diet assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        print("OpenAI Response:", response)
        tokens_used = response['usage']['total_tokens']
        print(f"Tokens used in this request: {tokens_used}")

        plan = response.choices[0].message["content"]

        plan_lines = plan.split("\n")
        diet_plan = ""
        diet_summary = ""

        current_section = None
        for line in plan_lines:
            if line.startswith("3. Three-Day Diet Plan"):
                current_section = "diet_plan"
            elif line.startswith("4. Diet Plan Summary"):
                current_section = "diet_summary"
            elif current_section:
                if current_section == "diet_plan":
                    diet_plan += line + "\n"
                elif current_section == "diet_summary":
                    diet_summary += line + "\n"

        return jsonify({
            "username": user_id,
            "diet_plan": diet_plan.strip(),
            "diet_summary": diet_summary.strip()
        })

        # OpenAI Error Handling:
    except openai.error.RateLimitError:
        print("OpenAI Rate Limit Error:", str(e))
        return error_response(429, "Rate limit exceeded, please try again later.")

    except openai.error.AuthenticationError:
        print("OpenAI Authentication Failed:", str(e))
        return error_response(401, "OpenAI authentication failed. Please check your API key.")

    except openai.error.InvalidRequestError as e:
        print("OpenAI Invalide Request:", str(e))
        return error_response(400, f"Invalid request to OpenAI: {str(e)}")

    except openai.error.OpenAIError as e:
        print("OpenAI General Error:", str(e))
        return error_response(500, f"OpenAI Error: {str(e)}")

    except Exception as e:
        print("General Error in generate_diet_plan:", str(e))
        return error_response(500, str(e))


# Validate Password


def is_valid_password(password):
    if len(password) < 8:
        return False
    if not (any(char.isdigit() for char in password) and
            any(char.isupper() for char in password) and
            any(char.islower() for char in password)):
        return False
    return True

# Users Signup POST Route


@main_blueprint.route("/users", methods=["POST"])
@cross_origin()
def signup():
    data = request.get_json()

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not all([username, email, password]):
        return error_response(400, "Missing Fields!")

    if not is_valid_password(password):
        return error_response(400, "Password must be at least 8 characters long, contain an uppercase letter, a lowercase letter, and a digit")

    if User.query.filter_by(username=username).first():
        return error_response(400, "Username already exists!")
    if User.query.filter_by(email=email).first():
        return error_response(400, "Email already exists!")

    new_user = User(
        username=data["username"],
        email=data["email"],
        age=data.get("age"),
        sex=data.get("sex"),
        weight=data.get("weight"),
        feet=data.get("feet"),
        inches=data.get("inches"),
        goals=data.get("goals"),
        days_per_week=data.get("days_per_week"),
        dietary_restrictions=data.get("dietary_restrictions")
    )
    new_user.set_password(data["password"])

    db.session.add(new_user)
    db.session.commit()

    access_token = create_access_token(
        identity=new_user.id, expires_delta=timedelta(weeks=1))

    return jsonify(
        access_token=access_token,
        user_id=new_user.id,
        username=new_user.username
    ), 201


# Get User Route
@main_blueprint.route("/users/<string:user_id>", methods=["GET"])
@jwt_required()
@cross_origin()
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "age": user.age,
        "sex": user.sex,
        "weight": user.weight,
        "feet": user.feet,
        "inches": user.inches,
        "goals": user.goals,
        "days_per_week": user.days_per_week,
        "dietary_restrictions": user.dietary_restrictions,
    })


# Update User Route


@main_blueprint.route("/users/<int:user_id>", methods=["PUT"])
@jwt_required()
@cross_origin()
def update_user(user_id):
    authenticated_user_id = int(get_jwt_identity())

    if user_id != authenticated_user_id:
        return error_response(403, "Unauthorized action!")

    user = User.query.get_or_404(user_id)
    data = request.get_json()

    if "old_password" in data and "new_password" in data:
        if not user.check_password(data["old_password"]):
            return error_response(400, "Current password is incorrect")

        if not is_valid_password(data["new_password"]):
            return error_response(400, "Password must be at least 8 characters long, contain an uppercase letter, a lowercase letter, and a digit")

        user.set_password(data["new_password"])

    new_username = data.get("username")
    if new_username and new_username != user.username:
        existing_user = User.query.filter_by(username=new_username).first()
        if existing_user:
            return error_response(400, "Username already exists!")

    new_email = data.get("email")
    if new_email and new_email != user.email:
        existing_user = User.query.filter_by(email=new_email).first()
        if existing_user:
            return error_response(400, "Email already exists!")

    if new_username:
        user.username = new_username
    if new_email:
        user.email = new_email

    user.age = data.get("age", user.age)
    user.sex = data.get("sex", user.sex)
    user.weight = data.get("weight", user.weight)
    user.feet = data.get("feet", user.feet)
    user.inches = data.get("inches", user.inches)
    user.goals = data.get("goals", user.goals)
    user.days_per_week = data.get("days_per_week", user.days_per_week)
    user.dietary_restrictions = data.get(
        "dietary_restrictions", user.dietary_restrictions)

    db.session.commit()
    return jsonify({"message": "User updated successfully!"})

# Users DELETE Route


@main_blueprint.route("/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
@cross_origin()
def delete_user(user_id):
    authenticated_user_id = int(get_jwt_identity())
    if user_id != authenticated_user_id:
        return error_response(403, "Unauthorized action!")

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted successfully!"})

# Users LOGIN Route


@main_blueprint.route("/login", methods=["POST"])
@cross_origin()
def login():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    print("Received email:", email)
    print("Received password:", password)

    if not all([email, password]):
        return error_response(400, "Missing fields!")

    user = User.query.filter_by(email=email).first()

    if user and user.check_password(password):
        access_token = create_access_token(
            identity=user.id, expires_delta=timedelta(weeks=1))

        response = jsonify(
            access_token=access_token,
            user_id=user.id,
            username=user.username
        ), 200
        return response


# Save Fitness Plan POST Route
@main_blueprint.route("/save_fitness_plan", methods=["POST"])
@jwt_required()
@cross_origin()
def save_fitness_plan():
    try:
        data = request.get_json()
        user_id = data["user_id"]
        user = User.query.get(user_id)

        if user is None:
            return error_response(404, "User not found")

        workout_routine = data["workout_routine"]
        workout_summary = data["workout_summary"]
        plan_name = data.get("plan_name")

        new_fitness_plan = SavedFitnessPlan(
            user=user,
            workout_routine=workout_routine,
            workout_summary=workout_summary,
            plan_name=plan_name
        )

        db.session.add(new_fitness_plan)
        db.session.commit()

        return jsonify({"message": "Fitness plan saved successfully!"}), 201

    except Exception as e:
        return error_response(500, str(e))


# Save Diet Plan POST Route
@main_blueprint.route("/save_diet_plan", methods=["POST"])
@jwt_required()
@cross_origin()
def save_diet_plan():
    try:
        data = request.get_json()
        user_id = data["user_id"]
        user = User.query.get(user_id)

        if user is None:
            return error_response(404, "User not found")

        diet_plan = data["diet_plan"]
        diet_summary = data["diet_summary"]
        plan_name = data.get("plan_name")

        new_diet_plan = SavedDietPlan(
            user=user,
            diet_plan=diet_plan,
            diet_summary=diet_summary,
            plan_name=plan_name
        )

        db.session.add(new_diet_plan)
        db.session.commit()

        return jsonify({"message": "Diet plan saved successfully!"}), 201

    except Exception as e:
        return error_response(500, str(e))


# Delete Saved Fitness Plan Route
@main_blueprint.route("/my_fitness_plans/<int:plan_id>", methods=["DELETE"])
@jwt_required()
@cross_origin()
def delete_fitness_plan(plan_id):
    try:
        authenticated_user_id = int(get_jwt_identity())

        plan = SavedFitnessPlan.query.get_or_404(
            plan_id)  # Use SavedFitnessPlan model

        if plan.user_id != authenticated_user_id:
            return error_response(403, "Unauthorized action!")

        db.session.delete(plan)
        db.session.commit()

        return jsonify({"message": "Fitness plan deleted successfully!"})

    except Exception as e:
        return error_response(500, str(e))

# Delete Saved Diet Plan Route


@main_blueprint.route("/my_diet_plans/<int:plan_id>", methods=["DELETE"])
@jwt_required()
@cross_origin()
def delete_diet_plan(plan_id):
    try:
        authenticated_user_id = int(get_jwt_identity())

        plan = SavedDietPlan.query.get_or_404(
            plan_id)  # Use SavedDietPlan model

        if plan.user_id != authenticated_user_id:
            return error_response(403, "Unauthorized action!")

        db.session.delete(plan)
        db.session.commit()

        return jsonify({"message": "Diet plan deleted successfully!"})

    except Exception as e:
        return error_response(500, str(e))


# GET ALL FITNESS PLANS FOR USER
@main_blueprint.route("/my_fitness_plans", methods=["GET"])
@jwt_required()
@cross_origin()
def get_user_fitness_plans():
    try:
        user_id = get_jwt_identity()

        fitness_plans = SavedFitnessPlan.query.filter_by(
            user_id=user_id).all()

        fitness_plans_list = [{
            "id": plan.id,
            "workout_routine": plan.workout_routine,
            "workout_summary": plan.workout_summary,
            "plan_name": plan.plan_name,
            "created_at": plan.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } for plan in fitness_plans]

        return jsonify(fitness_plans_list)

    except Exception as e:
        return error_response(500, str(e))


# GET ALL DIET PLANS FOR USER
@main_blueprint.route("/my_diet_plans", methods=["GET"])
@jwt_required()
@cross_origin()
def get_user_diet_plans():
    try:
        user_id = get_jwt_identity()

        diet_plans = SavedDietPlan.query.filter_by(
            user_id=user_id).all()

        diet_plans_list = [{
            "id": plan.id,
            "diet_plan": plan.diet_plan,
            "diet_summary": plan.diet_summary,
            "plan_name": plan.plan_name,
            "created_at": plan.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } for plan in diet_plans]

        return jsonify(diet_plans_list)

    except Exception as e:
        return error_response(500, str(e))


# GET FITNESS PLAN BY ID FOR USER
@main_blueprint.route("/my_fitness_plans/<int:plan_id>", methods=["GET"])
@jwt_required()
@cross_origin()
def get_single_user_fitness_plan(plan_id):
    try:
        user_id = get_jwt_identity()

        plan = SavedFitnessPlan.query.filter_by(
            id=plan_id, user_id=user_id).first()

        if not plan:
            return error_response(404, "Fitness plan not found or unauthorized.")

        plan_details = {
            "id": plan.id,
            "workout_routine": plan.workout_routine,
            "workout_summary": plan.workout_summary,
            "plan_name": plan.plan_name,
            "created_at": plan.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(plan_details)

    except Exception as e:
        return error_response(500, str(e))


# GET DIET PLAN BY ID FOR USER
@main_blueprint.route("/my_diet_plans/<int:plan_id>", methods=["GET"])
@jwt_required()
@cross_origin()
def get_single_user_diet_plan(plan_id):
    try:
        user_id = get_jwt_identity()

        plan = SavedDietPlan.query.filter_by(
            id=plan_id, user_id=user_id).first()

        if not plan:
            return error_response(404, "Diet plan not found or unauthorized.")

        plan_details = {
            "id": plan.id,
            "diet_plan": plan.diet_plan,
            "diet_summary": plan.diet_summary,
            "plan_name": plan.plan_name,
            "created_at": plan.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(plan_details)

    except Exception as e:
        return error_response(500, str(e))

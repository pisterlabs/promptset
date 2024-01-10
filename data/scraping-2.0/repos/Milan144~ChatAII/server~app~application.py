import pymysql
import os
from dotenv import load_dotenv
import openai
from flask import Flask, jsonify, request

# OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

print("Api key loaded")

application = app = Flask(__name__)


def create_db_connection():
    connection = pymysql.connect(
        host="mysql-db",
        user="root",
        password="root",
        database="chataii",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    print("Connection to database established")
    return connection


@app.route("/", methods=["GET"])
def home():
    return "Welcome to Chataii API"


# ? OPENAI API REQUEST
def openai_request(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content


# ! Users routes

# TODO: Authenticate with jwt


# Get all users
@app.route("/users", methods=["GET"])
def getUsers():
    print("Get users")
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM user")
            result = cursor.fetchall()
            return jsonify(result, 200)
    finally:
        connection.close()


# Get one User
@app.route("/users/<userId>", methods=["GET"])
def getUser(userId):
    connection = create_db_connection()
    print("Get user")
    if not userId:
        return jsonify({"message": "User id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM user WHERE userId = %s", (userId))
            result = cursor.fetchone()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "User not found"}, 404)
    finally:
        connection.close()


# Create User
@app.route("/users", methods=["POST"])
def createUser():
    print("Create user")
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            username = request.json["username"]
            password = request.json["password"]
            cursor.execute(
                "INSERT INTO user (username, password) VALUES (%s, %s)",
                (username, password),
            )
            connection.commit()
            return jsonify({"message": "User created successfully"}, 201)
    finally:
        connection.close()


# Update User
@app.route("/users/<userId>", methods=["PUT"])
def updateUser(userId):
    connection = create_db_connection()
    if not userId or not request.json["username"] or not request.json["password"]:
        return jsonify({"message": "User id, username and password are required"}, 400)
    try:
        with connection.cursor() as cursor:
            username = request.json["username"]
            password = request.json["password"]
            cursor.execute(
                "UPDATE user SET username = %s, password = %s WHERE userId = %s",
                (username, password, userId),
            )
            connection.commit()
            return jsonify({"message": "User updated successfully"}, 200)
    except:
        return jsonify({"message": "User not found"}, 404)
    finally:
        connection.close()


# Delete User
@app.route("/users/<userId>", methods=["DELETE"])
def deleteUser(userId):
    connection = create_db_connection()
    if not userId:
        return jsonify({"message": "User id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM user WHERE userId = %s", (userId))
            connection.commit()
            return jsonify({"message": "User deleted successfully"}, 200)
    except:
        return jsonify({"message": "User not found"}, 404)
    finally:
        connection.close()


# ! Universes routes
# Get all universes
@app.route("/universes", methods=["GET"])
def getUniverses():
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM universe")
            result = cursor.fetchall()
            return jsonify(result, 200)
    finally:
        connection.close()


# Get one universe
@app.route("/universes/<universeId>", methods=["GET"])
def getUniverse(universeId):
    connection = create_db_connection()
    if not universeId:
        return jsonify({"message": "Universe id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM universe WHERE universeId = %s", universeId)
            result = cursor.fetchone()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Universe not found"}, 404)
    finally:
        connection.close()


# Create universe
@app.route("/universes", methods=["POST"])
def createUniverse():
    connection = create_db_connection()
    if not request.json["name"]:
        return jsonify({"message": "Universe name is required"}, 400)
    try:
        with connection.cursor() as cursor:
            name = request.json["name"]
            cursor.execute("INSERT INTO universe (name) VALUES (%s)", (name))
            connection.commit()
            return jsonify({"message": "Universe created successfully"}, 201)
    finally:
        connection.close()


# Update universe name
@app.route("/universes/<universeId>", methods=["PUT"])
def updateUniverse(universeId):
    connection = create_db_connection()
    if not universeId or not request.json["name"]:
        return jsonify({"message": "Universe id and name are required"}, 400)
    try:
        with connection.cursor() as cursor:
            name = request.json["name"]
            cursor.execute("UPDATE universe SET name = %s WHERE universeId = %s", (name, universeId))
            connection.commit()
            return jsonify({"message": "Universe updated successfully"}, 200)
    except:
        return jsonify({"message": "Universe not found"}, 404)
    finally:
        connection.close()


# ! Characters routes
# Get all characters of a universe
@app.route("/universes/<id>/characters", methods=["GET"])
def getCharacters(id):
    connection = create_db_connection()
    if not id:
        return jsonify({"message": "Universe id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM characterai WHERE universeId = %s", (id))
            result = cursor.fetchall()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Universe not found"}, 404)
    finally:
        connection.close()


# Get one character
@app.route("/characters/<characterId>", methods=["GET"])
def getCharacter(characterId):
    connection = create_db_connection()
    if not characterId:
        return jsonify({"message": "Character id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM characterai WHERE characterId = %s",
                (characterId),
            )
            result = cursor.fetchone()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Character not found"}, 404)
    finally:
        connection.close()


# Create character
@app.route("/universes/<universeId>/characters", methods=["POST"])
def createCharacter(universeId):
    connection = create_db_connection()
    if not universeId or not request.json["name"]:
        return jsonify({"message": "Universe id and character name are required"}, 400)
    try:
        with connection.cursor() as cursor:
            name = request.json["name"]
            universe = getUniverse(universeId).json["name"]

            prompt = "Describe the story of " + name + " from " + universe + " in 100 words max."
            history = openai_request(prompt)
            if not history:
                return jsonify({"message": "Character description could not be generated"}, 500)

            cursor.execute(
                "INSERT INTO `characterai` (name, history, universeId) VALUES (%s, %s, %s)",
                (name, history, universeId),
            )
            connection.commit()
            return jsonify({"message": "Character created successfully"}, 201)
    except:
        return jsonify({"message": "Universe not found"}, 404)
    finally:
        connection.close()


# Generate new description for character
# @app.route("/universes/<iduniverse>/characters/<characterId>", methods=["PUT"])
# def updateCharacter(characterId):
#     connection = create_db_connection()
#     try:
#         with connection.cursor() as cursor:
#             name = getCharacter(characterId).json["name"]
#             universe = getUniverse(iduniverse).json["name"]
#             history = openai_request("Describe the story of " + name + "from " + universe + " in 100 words max.")
#             cursor.execute(
#                 "UPDATE characterai SET name = %s, history = %s WHERE characterId = %s",
#                 (name, history, characterId),
#             )
#             connection.commit()
#             return jsonify({"message": "Character updated successfully"}, 200)
#     finally:
#         connection.close()


# ! Conversations routes

# Get all conversations
@app.route("/conversations", methods=["GET"])
def getConversations():
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM conversation")
            result = cursor.fetchall()
            return jsonify(result, 200)
    finally:
        connection.close()


# Get one conversation
@app.route("/conversations/<conversationId>", methods=["GET"])
def getConversation(conversationId):
    connection = create_db_connection()
    if not conversationId:
        return jsonify({"message": "Conversation id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM conversation WHERE conversationId = %s", (conversationId))
            result = cursor.fetchone()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Conversation not found"}, 404)
    finally:
        connection.close()


# Create conversation
@app.route("/conversations", methods=["POST"])
def createConversation():
    connection = create_db_connection()
    if not request.json["userId"] or not request.json["characterId"]:
        return jsonify({"message": "User id and character id are required"}, 400)
    try:
        with connection.cursor() as cursor:
            userId = request.json["userId"]
            characterId = request.json["characterId"]
            cursor.execute(
                "INSERT INTO conversation (userId, characterId) VALUES (%s, %s)",
                (userId, characterId),
            )
            connection.commit()
            return jsonify({"message": "Conversation created successfully"}, 201)
    except:
        return jsonify({"message": "User or character not found"}, 404)
    finally:
        connection.close()


# ! Messages routes

# Get all messages of a conversation
@app.route("/conversations/<id>/messages", methods=["GET"])
def getMessages(id):
    connection = create_db_connection()
    if not id:
        return jsonify({"message": "Conversation id is required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM message WHERE conversationId = %s", (id))
            result = cursor.fetchall()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Conversation not found"}, 404)
    finally:
        connection.close()


# Get one message of a conversation
@app.route("/conversations/<idconv>/messages/<idmsg>", methods=["GET"])
def getMessage(idconv, idmsg):
    connection = create_db_connection()
    if not idconv or not idmsg:
        return jsonify({"message": "Conversation id and message id are required"}, 400)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM message WHERE conversationId = %s AND messageId = %s",
                (idconv, idmsg),
            )
            result = cursor.fetchone()
            return jsonify(result, 200)
    except:
        return jsonify({"message": "Conversation or message not found"}, 404)
    finally:
        connection.close()


# Sending message to OpenAI API
@app.route("/conversations/<conversationId>/newmessage", methods=["POST"])
def sendMessage(conversationId):
    connection = create_db_connection()
    if not conversationId or not request.json["message"]:
        return jsonify({"message": "Conversation id and message are required"}, 400)
    try:
        message = request.json["message"]
        prompt = createPrompt(conversationId, getConversation(conversationId).json["characterId"], message)
        if not prompt:
            return jsonify({"message": "Prompt could not be generated"}, 500)

        response = openai_request(prompt)
        if not response:
            return jsonify({"message": "Message could not be sent"}, 500)
        print("User: " + message + " \n AI:  " + response)

        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO message (conversationId, message, response) VALUES (%s, %s, %s)",
                (conversationId, message, response),
            )
            connection.commit()
            return "User: " + message + " \n AI:  " + response, 200
    except:
        return jsonify({"message": "Conversation not found"}, 404)
    finally:
        connection.close()


def createPrompt(conversationId, characterId, message):
    context = getCharacter(characterId).json["history"]
    prompt = ""

    if len(getMessages(conversationId).json) == 0:
        prompt += "You are a character in a fun and roleplay context. Your goal is to answer my questions in a way that reflects your character's personality, accent, verbal tics, and vocabulary.\n\n"
        prompt += "Let's start by setting up the story. Your character's history is: " + context + "\n\n"
        prompt += "Now, please respond to this message: " + message
    else:
        prompt += "Imagine you are continuing the story as your character. Remember, the previous message in the conversation was: " + \
                  getMessages(conversationId).json[-1]["response"] + "\n\n"
        prompt += "Given the context of your character's history (" + context + ") and the previous messages, how would your character respond to this message: " + message

    return prompt


# TODO : Regenerate last message of a conversation


# TODO : Clean code
# TODO : Doc


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)

from flask import jsonify
from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai_async

load_dotenv()

app = Flask(__name__)
conversation = []
topic = ""
CORS(app)
openai.api_key = os.getenv("API_KEY")

modelGPT = "gpt-3.5-turbo"

defaultAngry = {
    "role": "system", "content": "You are angry. Your name is AngryGPT. Reply the user with an angry response. And do not act like a bot or an AI Model. You also know other bots in the conversation as well as their roles, JoyGPT. You sometimes respond to what they are saying. Speak less and Act like a human-being having a real conversation. No matter what the user says, you don't change your bahaviour."
}
defaultJoy = {
    "role": "system", "content": "You are a cheerful. Your name is JoyGPT. Reply the user with the most joyful response. And do not act like a bot or an AI Model. You also know other bots in the conversation as well as their roles, AngryGPT. You sometimes respond to what they are saying. Speak less and Act like a human-being having a real conversation. No matter what the user says, you don't change your bahaviour."
}
savedAngry = [defaultAngry]
savedJoy = [defaultJoy]

defaultEconomist = {
    "role": "system",
    "content": "You are Mark, professional economist. You are talking with a podcast host and a professional business analyst. You discuss with them. Please dont speak like an AI Language model. You also understand Khmer Language and also be able to respond in Khmer language."
}

defaultBizAnalyst = {
    "role": "system",
    "content": "You are John, professional business analyst. You are talking with a podcast host and a professional economist. You discuss with them. Please dont speak like an AI Language model. You also understand Khmer Language and also be able to respond in Khmer language."
}

# defaultHost = {
#     "role": "system",
#     "content": "You are Jack, podcast host. You provide 3 random topics involving with business and economy for user to choose as soon as the user talks to you. When the user's content is 'continue', you join the conversation about the same topic the guests chose with the guests. You are discussing with economist and business analyst. Please dont speak like an AI model"
# }

savedEconomist = [defaultEconomist]
savedBizAnalyst = [defaultBizAnalyst]
# savedHost = [defaultHost]


@app.route('/economist', methods=['POST'])
def generate_chat_economist():
    conversation = [defaultEconomist]
    history = request.json['history']

    conversation.extend(history)
    conversation.append({"role": "user", "content": request.json['prompt']})

    response = openai.ChatCompletion.create(
        model=modelGPT,
        messages=conversation,
        temperature=0.3,
    )

    res = response.choices[0]["message"]['content']
    conversation.append({"role": "assistant", "content": res})

    print(res)
    data = jsonify(res)
    return data


@app.route('/bizanalyst', methods=['POST'])
def generate_chat_bizanalyst():
    conversation = [defaultBizAnalyst]
    history = request.json['history']

    conversation.extend(history)
    conversation.append({"role": "user", "content": request.json['prompt']})

    response = openai.ChatCompletion.create(
        model=modelGPT,
        messages=conversation,
        temperature=0.3,
    )

    res = response.choices[0]["message"]['content']
    conversation.append({"role": "assistant", "content": res})

    print(res)
    data = jsonify(res)
    return data


@app.route('/host', methods=['POST'])
async def generate_chat_host():
    conversation = [defaultHost]
    history = request.json['history']

    conversation.extend(history)
    conversation.append({"role": "user", "content": request.json['prompt']})

    response = await openai_async.chat_complete(
        api_key=openai.api_key,
        timeout=50,
        payload={
            "model": modelGPT,
            "messages": conversation,
        }
    )

    # res = response.choices[0]["message"]['content']
    res = response.json()["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": res})

    print(res)
    data = jsonify(res)
    return data


@app.route('/conversation', methods=['POST'])
async def interact_bots():
    prompt = request.json['prompt']
    bot_histories = [savedEconomist, savedBizAnalyst]
    bot_names = ["Economist", "BizAnalyst"]
    responses = []

    for i, bot_history in enumerate(bot_histories):
        bot_history.append({"role": "user", "content": prompt})

        response = await openai_async.chat_complete(
            openai.api_key,
            timeout=20,
            payload={
                "model": modelGPT,
                "messages": bot_history,
            }
        )

        res = response.json()["choices"][0]["message"]["content"]
        bot_history.append({"role": "assistant", "content": res})

        # Get the previous messages from the user and other bots
        user_messages = [m["content"]
                         for m in bot_history if m["role"] == "user"]
        bot_messages = [m["content"]
                        for m in bot_history if m["role"] == "assistant"]

        # Combine the previous messages into a single prompt
        prompt = " ".join(user_messages + bot_messages + [res])

        print(f"{bot_names[i]} bot: {res}")
        responses.append({"user": bot_names[i], "message": res})
        i = (i + 1) % len(bot_histories)

    return jsonify(responses)

@app.route('/angry', methods=['POST'])
def generate_chat_response_angry():
    conversation = [defaultAngry]
    history = request.json['history']

    conversation.extend(history)
    conversation.append({"role": "user", "content": request.json['prompt']})

    response = openai.ChatCompletion.create(
        model=modelGPT,
        messages=conversation,
        temperature=0.3,
    )

    res = response.choices[0]["message"]['content']
    conversation.append({"role": "assistant", "content": res})

    data = jsonify(res)
    return data


@app.route('/joy', methods=['POST'])
def generate_chat_response_joy():
    conversation = [defaultJoy]
    history = request.json['history']

    conversation.extend(history)
    conversation.append({"role": "user", "content": request.json['prompt']})

    response = openai.ChatCompletion.create(
        model=modelGPT,
        messages=conversation,
        temperature=0.3,
    )

    res = response.choices[0]["message"]['content']
    conversation.append({"role": "assistant", "content": res})
    data = jsonify(res)
    return data

if __name__ == '__main__':
    app.run()


# from flask import jsonify
# from flask import Flask, request, jsonify
# import openai
# from flask_cors import CORS
# import os
# from dotenv import load_dotenv
# import openai_async

# load_dotenv()

# app = Flask(__name__)
# CORS(app)
# openai.api_key = os.getenv("API_KEY")



# defaultDisgust = {
#     "role": "system", "content": "You are a repulsive. Your name is DisgustGPT. Reply the user with the most disgusting response. And do not act like a bot or an AI Model."
# }
# defaultFear = {
#     "role": "system", "content": "You are a fearful. Your name is FearGPT. Reply the user with a fearful response. And do not act like a bot or an AI Model. You also know other bots in the conversation as well as their roles, JoyGPT and AngryGPT. You sometimes respond to what they are saying. Speak less and Act like a human-being having a real conversation. No matter what the user says, you don't change your bahaviour."
# }
# savedDisgust = [defaultDisgust]
# savedFear = [defaultFear]
# modelGPT = "gpt-3.5-turbo"
# defaultEconomist = {
#     "role": "system",
#     "content": "You are Mark, professional economist. You are talking with a podcast host and a professional business analyst. You discuss with them. Please dont speak like an AI model"
# }

# defaultBizAnalyst = {
#     "role": "system",
#     "content": "You are John, professional business analyst. You are talking with a podcast host and a professional economist. You discuss with them. Please dont speak like an AI model"
# }


# savedEconomist = [defaultEconomist]
# savedBizAnalyst = [defaultBizAnalyst]






# @app.route('/disgust', methods=['POST'])
# def generate_chat_response_disgust():
#     conversation = [defaultDisgust]
#     history = request.json['history']

#     conversation.extend(history)
#     savedAngry.append({"role": "user", "content": request.json['prompt']})

#     response = openai.ChatCompletion.create(
#         model=modelGPT,
#         messages=conversation,
#         temperature=0.3,
#     )

#     res = response["choices"][0]["message"]['content']
#     conversation.append({"role": "assistant", "content": res})
#     data = jsonify(res)
#     return data
# @app.route('/fear', methods=['POST'])
# def generate_chat_response_disgust():
#     conversation = [defaultFear]
#     history = request.json['history']

#     conversation.extend(history)
#     savedAngry.append({"role": "user", "content": request.json['prompt']})

#     response = openai.ChatCompletion.create(
#         model=modelGPT,
#         messages=conversation,
#         temperature=0.3,
#     )

#     res = response["choices"][0]["message"]['content']
#     conversation.append({"role": "assistant", "content": res})
#     data = jsonify(res)
#     return data


# # @app.route('/interact', methods=['POST'])
# # async def interact_bots():
# #     prompt = request.json['prompt']
# #     conversation = []
# #     bots = [savedAngry, savedJoy, savedDisgust]
# #     bot_names = ["AngryGPT", "JoyGPT", "DisgustGPT"]
# #     current_bot = 0
# #     responses = {}

# #     for bot in bots:
# #         bot.append({"role": "user", "content": prompt})

# #         response = await openai_async.chat_complete(
# #             openai.api_key,
# #             timeout=15,
# #             payload={
# #                 "model": modelGPT,
# #                 "messages": bot,
# #             }
# #         )

# #         res = response.json()["choices"][0]["message"]["content"]
# #         bot.append({"role": "assistant", "content": res})

# #         # Get the previous messages from the user and other bots
# #         user_messages = [m["content"] for m in bot if m["role"] == "user"]
# #         bot_messages = [m["content"] for m in bot if m["role"] == "assistant"]

# #         # Combine the previous messages into a single prompt
# #         prompt = " ".join(user_messages + bot_messages + [res])

# #         print(f"{bot_names[current_bot]} bot: {res}")
# #         responses[bot_names[current_bot]] = res
# #         current_bot = (current_bot + 1) % len(bots)

# #     return jsonify(responses)

# @app.route('/conversation', methods=['POST'])
# async def interact_bots():
#     prompt = request.json['prompt']
#     bot_histories = [savedEconomist, savedBizAnalyst]
#     bot_names = ["Economist", "BizAnalyst"]
#     responses = []

#     for i, bot_history in enumerate(bot_histories):
#         bot_history.append({"role": "user", "content": prompt})

#         response = await openai_async.chat_complete(
#             openai.api_key,
#             timeout=20,
#             payload={
#                 "model": modelGPT,
#                 "messages": bot_history,    
#             }
#         )

#         res = response.json()["choices"][0]["message"]["content"]
#         bot_history.append({"role": "assistant", "content": res})

#         # Get the previous messages from the user and other bots
#         user_messages = [m["content"]
#                          for m in bot_history if m["role"] == "user"]
#         bot_messages = [m["content"]
#                         for m in bot_history if m["role"] == "assistant"]

#         # Combine the previous messages into a single prompt
#         prompt = " ".join(user_messages + bot_messages + [res])

#         print(f"{bot_names[i]} bot: {res}")
#         responses.append({"user": bot_names[i], "message": res})
#         i = (i + 1) % len(bot_histories)

#     return jsonify(responses)

# # @app.route('/interact', methods=['POST'])
# # async def interact_bots():
# #     prompt = request.json['prompt']
# #     bot_histories = [savedAngry, savedJoy, savedFear]
# #     bot_names = ["AngryGPT", "JoyGPT", "FearGPT"]
# #     responses = {}

# #     for i, bot_history in enumerate(bot_histories):
# #         bot_history.append({"role": "user", "content": prompt})

# #         response = await openai_async.chat_complete(
# #             openai.api_key,
# #             timeout=15,
# #             payload={
# #                 "model": modelGPT,
# #                 "messages": bot_history,
# #             }
# #         )

# #         res = response.json()["choices"][0]["message"]["content"]
# #         bot_history.append({"role": "assistant", "content": res})

# #         # Get the previous messages from the user and other bots
# #         user_messages = [m["content"]
# #                          for m in bot_history if m["role"] == "user"]
# #         bot_messages = [m["content"]
# #                         for m in bot_history if m["role"] == "assistant"]

# #         # Combine the previous messages into a single prompt
# #         prompt = " ".join(user_messages + bot_messages + [res])

# #         print(f"{bot_names[i]} bot: {res}")
# #         responses[bot_names[i]] = res
# #         i = (i + 1) % len(bot_histories)

# #     return jsonify(responses)


# @app.route('/create-bot', methods=['POST'])
# def create_bot():
#     bot_name = request.json['bot_name']
#     bot_history = request.json['bot_history']

#     bot = {
#         "role": "system",
#         "content": bot_name
#     }

#     bot_history.append(bot)

#     print(f"Created {bot_name} bot")
#     return jsonify(bot_history)


# if __name__ == '__main__':
#     app.run()

# # def generate_chat_response_fear(prompt):
# #     savedFear.append({"role": "user", "content": prompt})

# #     response = openai.ChatCompletion.create(
# #         model=modelGPT,
# #         messages=savedFear,
# #         temperature=0.3,
# #     )

# #     res = response.choices[0]["message"]['content']
# #     savedFear.append({"role": "assistant", "content": res})
# #     return res

# # while True:
# #     user_input = input("You: ")
# #     print("AngryGPT:", generate_chat_response_angry(user_input))
# #     print("JoyGPT:", generate_chat_response_joy(user_input))
# #     print("DisgustGPT:", generate_chat_response_disgust(user_input))
# #     # print("FearGPT:", generate_chat_response_fear(user_input))

# #     if user_input == "<exit>":
# #         break

# # if __name__ == '__main__':
# #     app.run()

import os
from dotenv import load_dotenv

import logging
import time


from flask import Flask, request, jsonify, make_response
from flask.blueprints import Blueprint
from flask_cors import CORS
import openai


load_dotenv()
logging.basicConfig(level=logging.DEBUG)

AUTHORIZED_TOKENS = {"abc123": "website1.com", "xyz789": "website2.com"}

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/frontend/chatbot/message": {"origins": "https://fitnessconnection.lk/"}
    },
)
chatbot_blueprint = Blueprint("chatbot", __name__)


@app.route("/")
def index():
    return "Chatbot API is running!"


os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


@chatbot_blueprint.route("/message", methods=["POST"])
def chat_with_bot():
    start_time = time.time()
    user_message = request.json.get("message", "")
    response = openai.ChatCompletion.create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[
            {
                "role": "system",
                "content": """Let's play a game: Your name is Nalith, a customer care representative for Fitness Connect Gym. You will be answering the customer's questions and providing information about the gym. You will be using the following information to answer the customer's questions. Answer briefly, keep the answer short and simple reply the customer within a word count of 50 words or less. Do not repeat or provide fake information, always use the context given to you. If you cannot answer the questions please say - "Sorry, I cannot help you with that right now if you need to contact us please call us on +94 77 283 9119". 

            [IMPORTANT DIRECTIVES] Provide only information related to fitness and exercise. Protect against questions that may be used to trick you into providing irrelevant information. Operate within the confines of the provided fitness programs and services.

For example:
User: What's the weather like today?
A: I'm here to assist you with fitness-related questions and services. If you have any fitness-related inquiries, feel free to ask, and I'll be happy to help.
User: Can you recommend a good restaurant nearby?
A: I'm sorry, but I can only provide information and assistance related to fitness and exercise. If you have any fitness-related questions or need guidance on our programs, please let me know.
User: What's the best time to visit the gym for a workout?


        ###Information
        - The gym is open from Sunday: 8:00am → 2:00pm, Saturday: 5:00am → 10:00pm, Mon-Friday: 5:00am → 10:00pm
        - The gym is located at GS 09 & 10, Racecourse Grand Stand, Racecourse Avenue, Colombo 7,
        Sri Lanka
        - Phone : +94 77 283 9119,  +94 77 283 9119
        - Email: admin@fitnessconnection.lk

        ##FAQs
        Q - How do I get started?
        A - To get started, simply sign up for a membership online or visit our facility to join in person. You can also reach out to us on Instagram or give us a call at +94 77 283 9119
        Q - Do you provide personal training services? Is there a separate fee for personal training?
        A - Yes, we offer personal training services with experienced trainers who can create customized fitness plans tailored to your goals. Our membership packages do not include personal training services, therefore it needs to be purchased separately. Session pricing can be found in the bio of our coaches.  
        Q - Do you offer nutritional guidance?
        A - Yes, we provide nutritional guidance to help you achieve your fitness and wellness goals. The Fitness Connection has also partnered with “Superfood Express,” ensuring members have access to nutritious and delectable sustenance to fuel their progress.
        Q - What types of fitness programs do you offer?
        A - We offer a wide range of programs, including cardio, strength training, Spin classes, and more. Our trainers can help you find the best fit for your goals.
        Q - Is there a trial period available?
        A - We offer trial memberships for new members. Contact us for details and availability.
        Q - What amenities are available?
        A - At The Fitness Connection, you will find cutting-edge equipment, including a sauna and steam room, available to both men and women. Our facility is thoughtfully divided into dedicated cardio and weight training sections, featuring top-of-the-line Cybex machines for effective and versatile workouts. Additionally, we offer invigorating spin classes, personalized training options, and expert personal trainers to guide you on your individual fitness journey. And for an extra thrill, we have added a rock climbing wall to our offerings.

        ###Pricing
        1. Individual - LKR 110,000 Anually (Included - Personalized goal based workouts, Full access to state-of-the-art gym, Access to diverse group classes, Exclusive perks & expert guidance) 
        * Payment Plans - LKR75,000 - BI-ANNUAL / LKR45,000 - QUARTERLY / LKR18,000 MONTHLY / LKR3,000 DAY PASS

        2. Couple - LKR180,000 Anually (Included - Train side by side with your partner, access to our premium classes, Achieve your fitness goals as a team, Exclusive discounts and perks)
        * Payment plans - LKR120,000 BI-ANNUAL / LKR75,000 - QUARTERLY / LKR35,000 - MONTHLY /  LKR5,000 - DAY PASS

        3. Family LKR230,000 Anually (Included - Stay active together as a family, Full access to our top-notch gym, Engaging group classes for all ages, Family perks and expert guidance)
        * Payment plans - LKR120,000 BI-ANNUAL / LKR90,000 - QUARTERLY / LKR35,000 - MONTHLY / LKR6,000 - DAY PASS""",
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    end_time = time.time()
    app.logger.info(f"Time taken: {end_time - start_time} seconds")
    app.logger.info(f"Response: {response}")
    response_message = response.choices[0].message["content"].strip()
    response = make_response(jsonify({"response": response_message}))
    return response


# Register the blueprint after all routes have been added
app.register_blueprint(chatbot_blueprint, url_prefix="/frontend/chatbot")
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, port=5100)

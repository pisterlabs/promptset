# from fitbit_user import FitbitUser
# import openai
# from flask import Flask
# from fitbit_user import FitbitUser
# import os

# app = Flask(__name__)

# FITBIT_CLIENT_ID = os.getenv('FITBIT_CLIENT_ID', '23QZ6Y')
# FITBIT_CLIENT_SECRET = os.getenv('FITBIT_CLIENT_SECRET', '0ab29d7e9a448844b4bfb0c889e56cd9')
# FITBIT_AUTH_URL = 'https://www.fitbit.com/oauth2/authorize'
# FITBIT_TOKEN_URL = 'https://api.fitbit.com/oauth2/token'
# FITBIT_REDIRECT_URL = os.getenv('FITBIT_REDIRECT_URI', 'https://fitbitdataincorporation.simonlizhm.repl.co/callback/fitbit')

# # Set up the Fitbit credentials
# app.config['FITBIT_CLIENT_ID'] = FITBIT_CLIENT_ID
# app.config['FITBIT_CLIENT_SECRET'] = FITBIT_CLIENT_SECRET
# app.config['FITBIT_REDIRECT_URL'] = FITBIT_REDIRECT_URL



# @app.route('/get_data')
# def get_data():
#     user = FitbitUser(app)
#     user_data = user.get_all_data()
#     # do something with user_data
#     return user_data

# # if __name__ == "__main__":
# user = get_data()

# # user = FitbitUser(app)
# # user_data = user.get_all_data()

# openai.api_key = "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"

# # app = Flask(__name__)
# # app.secret_key = 'your_secret_key'

# # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_data.db'
# # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# # db = SQLAlchemy(app)

# # login_manager = LoginManager()
# # login_manager.init_app(app)
# # login_manager.login_view = 'login'

# # fitbit_user = FitbitUser(app)

# def generate_recommendations(query):
#   user1 = FitbitUser(app)
#   user = user1.get_all_data()  # fix this

#   response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#       {
#         "role":
#         "system",
#         "content":
#         "As a health assistant, your task is to examine a user's query regarding their health and utilize the fitbit data provided, which includes various metrics such as steps_today, total_minutes_asleep, active_minutes, calories_burned, distance_covered, floors_climbed, sedentary_minutes, resting_heart_rate, time_in_bed, age, gender, height, and weight, to offer a pertinent medical suggestion. In cases where the data points do not indicate a definite recommendation, you should advise the user to seek advice from a healthcare professional. Please provide a thorough and comprehensive response."
#       },
#       {
#         "role": "user",
#         "content": "Why am I not seeing improvements in my running distance?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 5500, total_minutes_asleep: 330, active_minutes: 40, calories_burned: 1900, distance_covered: 2.5, floors_climbed: 15, sedentary_minutes: 980, resting_heart_rate: 70, time_in_bed: 360, age: 40, gender: 'Male', height: 180, weight: 75"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "You may not be seeing improvements because your activity levels are relatively low. Try to gradually increase your active minutes and focus on a mix of cardio and strength training exercises."
#       },
#       {
#         "role":
#         "user",
#         "content":
#         "I've been feeling unusually tired recently. Can you suggest why?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 3000, total_minutes_asleep: 200, active_minutes: 60, calories_burned: 2100, distance_covered: 1.5, floors_climbed: 5, sedentary_minutes: 1300, resting_heart_rate: 80, time_in_bed: 220, age: 50, gender: 'Female', height: 160, weight: 60"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "Your sleep duration is lower than the recommended 7-9 hours for adults, which could be causing you to feel tired. Try to improve your sleep routine and ensure you are getting enough rest."
#       },
#       {
#         "role":
#         "user",
#         "content":
#         "I am not losing weight despite regular exercise. What could be the reason?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 7000, total_minutes_asleep: 400, active_minutes: 120, calories_burned: 2300, distance_covered: 5.0, floors_climbed: 10, sedentary_minutes: 900, resting_heart_rate: 75, time_in_bed: 430, age: 28, gender: 'Male', height: 175, weight: 80"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "While you are exercising regularly, the calories burned amount to 2300, which might not create a significant caloric deficit if your calorie intake is high. Monitor your diet and try to maintain a balanced, low-calorie diet alongside regular exercise."
#       },
#       {
#         "role":
#         "user",
#         "content":
#         "I'm finding it hard to catch my breath during workouts. Could it be something serious?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 6000, total_minutes_asleep: 450, active_minutes: 100, calories_burned: 2200, distance_covered: 4.0, floors_climbed: 30, sedentary_minutes: 800, resting_heart_rate: 85, time_in_bed: 500, age: 38, gender: 'Female', height: 170, weight: 68"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "Your resting heart rate is a bit high and you're climbing a significant number of floors, which may put stress on your cardiovascular system. However, shortness of breath can also indicate underlying health issues. If this continues, please consult a healthcare professional."
#       },
#       {
#         "role": "user",
#         "content": "I'm always hungry. What can be the reason?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 5000, total_minutes_asleep: 350, active_minutes: 80, calories_burned: 2100, distance_covered: 3.0, floors_climbed: 15, sedentary_minutes: 850, resting_heart_rate: 72, time_in_bed: 380, age: 35, gender: 'Male', height: 185, weight: 82"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "If you're burning a significant amount of calories and are active, it's normal to feel hungry as your body is trying to replace the energy it lost. Ensure your meals are balanced with enough protein and fiber, which can help keep you feeling fuller for longer."
#       },
#       {
#         "role": "user",
#         "content": "Why do I feel so stressed all the time?"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "steps_today: 3000, total_minutes_asleep: 250, active_minutes: 50, calories_burned: 1800, distance_covered: 1.5, floors_climbed: 5, sedentary_minutes: 1400, resting_heart_rate: 78, time_in_bed: 300, age: 32, gender: 'Female', height: 160, weight: 60"
#       },
#       {
#         "role":
#         "assistant",
#         "content":
#         "Your data suggests a high number of sedentary minutes and low total sleep time. Both physical inactivity and lack of adequate sleep can contribute to stress. Try to incorporate more physical activity and relaxation techniques into your daily routine, and ensure you get enough sleep."
#       },
#       {
#         "role": "user",
#         "content": query
#       },
#     ])
#   return response['choices'][0]['message']['content']


# if __name__ == '__main__':
#   app.run(debug=True)
#   query = input("Please enter your health query: ")
#   print(generate_recommendations(query))


#Aryan code
from langchain import PromptTemplate, FewShotPromptTemplate
import openai
import os

# openai.api_key = "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"

os.environ["OPENAI_API_KEY"] = "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"

example_formatter_template = """
Query: {query}
Data: {data}
Recommendation: {recommendation}
"""

query_1 = "I've been feeling unusually tired recently. Can you suggest why?"
data_1 = "steps_today: 3000, total_minutes_asleep: 200, active_minutes: 60, calories_burned: 2100, distance_covered: 1.5, floors_climbed: 5, sedentary_minutes: 1300, resting_heart_rate: 80, time_in_bed: 220, age: 50, gender: 'Female', height: 160, weight: 60"
recommendation_1 = "Your sleep duration is lower than the recommended 7-9 hours for adults, which could be causing you to feel tired. Try to improve your sleep routine and ensure you are getting enough rest."

query_2 = "I am not losing weight despite regular exercise. What could be the reason?"
data_2 = "steps_today: 7000, total_minutes_asleep: 400, active_minutes: 120, calories_burned: 2300, distance_covered: 5.0, floors_climbed: 10, sedentary_minutes: 900, resting_heart_rate: 75, time_in_bed: 430, age: 28, gender: 'Male', height: 175, weight: 80"
recommendation_2 = "While you are exercising regularly, the calories burned amount to 2300, which might not create a significant caloric deficit if your calorie intake is high. Monitor your diet and try to maintain a balanced, low-calorie diet alongside regular exercise."

examples = [
    {
        "query": query_1,
        "data": data_1,
        "recommendation": recommendation_1,
    },
    {
        "query": query_2,
        "data": data_2,
        "recommendation": recommendation_2,
    },
]

from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

def generate_recommendation(query, data):
      
    example_prompt = PromptTemplate(
        input_variables=[
            "query",
            "data",
            "recommendation",
        ],
        template=example_formatter_template,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="As a health assistant, your task is to examine a user's query regarding their health and utilize the fitbit data provided to offer a pertinent medical suggestion. In cases where the data points do not indicate a definite recommendation, you should advise the user to seek advice from a healthcare professional. Please provide a thorough and comprehensive response.",
        suffix="Query: {query}\nData: {data}\nRecommendation:",
        input_variables=["query","data"],
        example_separator="\n\n",
    )

    final_prompt = few_shot_prompt.format(
        query=query.strip(),
        data=data.strip(),
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that uses Fitbit data to provide health recommendations.")
    human_message_prompt = HumanMessagePromptTemplate.from_template(final_prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat = ChatOpenAI(temperature=0.5)  # You can adjust the temperature parameter as needed
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run({"text": query})
    return response


if __name__ == "__main__":
    data = "steps_today: 5500, total_minutes_asleep: 330, active_minutes: 40, calories_burned: 1900, distance_covered: 2.5, floors_climbed: 15, sedentary_minutes: 980, resting_heart_rate: 70, time_in_bed: 360, age: 40, gender: 'Male', height: 180, weight: 75"
    query = input("Please enter your health query: ")
    recommendation = generate_recommendation(query, data)
    print(recommendation)
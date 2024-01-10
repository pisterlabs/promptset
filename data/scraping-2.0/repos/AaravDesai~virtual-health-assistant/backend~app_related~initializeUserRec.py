import sys
from flask import make_response
from bson import ObjectId
from db import db
import openai

def initializeUserRec_api(cookie):
    users_collection = db["users"]
    user = users_collection.find_one({"_id": ObjectId(cookie)})
    
    openai.api_type = "azure"
    openai.api_key = '9fd2d1a84636415685bcf7dc451040fb'
    openai.api_base = 'https://api.umgpt.umich.edu/azure-openai-api/ptu'
    openai.api_version = '2023-03-15-preview'
    
    conversation_history = [
        {"role": "system", "content": "You are a helpful virtual health assistant"},
    ]
    
    age = user['data']['age']
    gender = user['data']['gender']
    weight = user['data']['weight']
    height = user['data']['height']
    fitnessGoal = user['data']['fitnessGoal']
    experience = user['data']['experience']
    dietaryRestriction = user['data']['dietaryRestriction']
    numDays = user['data']['numDays']
    includeSupplements = user['data']['includeSupplements']
    
    while True:
        user_data = f"This is the profile of a {age} year old {gender} weighing {weight} and standing at {height} whose fitness goal is {fitnessGoal}. The individual is currently a {experience}. The individual has chosen to create a plan for {numDays} per week and has said {includeSupplements} to supplements."
        if (dietaryRestriction != "n/a"):
            user_data+="The person also has a {dietaryRestriction} dietary restriction."
        user_input="Create a diverse and stuctured diet & workout plan for the user using the responses provided. Make it extremely personalized and tailored to the number of workout days and goals that the user provides. Create a day by day plan for both the diet and workout (Base the workout on the fitness goal and the number of workout days a week indicated by the user and make it extremely accurate), and give the same format with the day by day and step by step Workout Routine and Diet. Include supplements in diet if user has said yes. Also provide with a note saying that you should consult a medical professional with the dosage of the supplement if indicated by the user that he wants a supplement. Use the format which I am providing below to base your answer on. It is IMPERATIVE that you follow the format exactly as far as syntax wise. Note only use the output format, the content is going to be determined by you. Don't use the example I gave you to come up with the recommendations but use the data about the person. \
            \
            **Day-by-Day Workout Routine**: \
                *Day 1 - Chest and Back* \
                \
                Warm-up with 10-15 minutes of cardio - Treadmill or cycling \
                1. Bench Press - 3 sets of 8-12 reps \
                2. Incline Dumbbell Press - 3 sets of 8-12 reps \
                3. Bent-over Rows - 3 sets of 8-12 reps \
                4. Lat Pulldowns - 3 sets of 8-12 reps \
                5. Finish with 10-15 minutes of stretching \
                \
                *Day 2 - Legs and Abs* \
                \
                Warm up with 10-15 minutes of cardio - Treadmill or cycling \
                1. Squats - 3 sets of 8-12 reps \
                2. Lunges - 3 sets of 8-12 reps per leg \
                3. Calf Raises - 3 sets of 15-20 reps \
                4. Planks - 3 sets of 30-60 seconds \
                5. Finish with 10-15 minutes of stretching \
                \
                *Rest on Day 3* \
                \
                *Day 4 - Arms and Shoulders* \
                \
                Warm up with 10-15 minutes of cardio - Treadmill or cycling \
                1. Bicep Curls - 3 sets of 8-12 reps \
                2. Tricep Dips - 3 sets of 8-12 reps \
                3. Shoulder Press - 3 sets of 8-12 reps \
                4. Finish with 10-15 minutes of stretching \
                \
                *Day 5 - Repeat of Day 1's routine* \
                \
                *Rest on Day 6* \
                \
                *Day 7 - Repeat of Day 2's routine*"" \
                \
                ***Diet Plan (Vegetarian)*** \
                \
                *Day 1:* \
                \
                - Breakfast: Oatmeal with chopped fruits \
                - Lunch: Chickpea salad with whole grain bread \
                - Dinner: Grilled cottage cheese with sautéed vegetables \
                - Snack: Guacamole and carrot sticks \
                \
                *Day 2:* \
                \
                - Breakfast: Greek yogurt with blueberries \
                - Lunch: Lentil soup with brown rice \
                - Dinner: Veggie stir fry with tofu \
                - Snack: Fruit smoothie \
                \
                *Day 3:* \
                \
                - Breakfast: Whole grain toast with avocado \
                - Lunch: Quinoa bowl with veggies and hummus \
                - Dinner: Stuffed bell peppers with bulgur and black beans \
                - Snack: A handful of nuts \
                \
                *Day 4:* \
                \
                - Breakfast: Scrambled tofu with spinach and tomatoes \
                - Lunch: Whole grain pasta with mushroom sauce \
                - Dinner: Vegetable curry with basmati rice \
                - Snack: A banana and a spoonful of almond butter \
                \
                ***Supplements*** \
                \
                Since you have indicated yes for supplements, here are a few you might consider: \
                \
                1. *Whey Protein*: Helps in muscle recovery and growth. \
                2. *Multivitamins*: Necessary for overall health. \
                3. *Creatine*: Helps in high intensity training. \
                \
                ***Note*: Please consult with a healthcare professional to determine the right dosage for your supplements based on your individual conditions and needs. * \
                \
                Now that you're good to go, I wish you the best of luck in your fitness journey! I hope you achieve your goal of gaining muscle mass and improving your overall fitness."
        final_input = user_input + user_data
        conversation_history.append({"role": "user", "content": final_input})
        response = openai.ChatCompletion.create(
            engine='gpt-4',
            messages=conversation_history
        )
        assistant_reply = response['choices'][0]['message']['content']
        
        index = assistant_reply.find("**Diet Plan")
        if not index:
            continue
        
        workout_plan = assistant_reply[:index]
        diet_plan = assistant_reply[index:]
        print(assistant_reply,workout_plan,diet_plan,file=sys.stderr)
        
        users_collection.update_one({"_id": ObjectId(cookie)},{'$set':{
            'workout_plan':workout_plan,
            'diet_plan':diet_plan
        }})
        break
    
    return make_response({"message":'Success'})
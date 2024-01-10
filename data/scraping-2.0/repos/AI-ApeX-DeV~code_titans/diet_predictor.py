from flask import Flask, render_template, request
import pyrebase
from flask_cors import CORS
import openai
app = Flask(__name__)
openai.api_key = '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'

CORS(app)
# Define your Firebase configuration data
config = {
  "apiKey": "AIzaSyAvYvSqBoQzCUDK2oloq79JhPJGTw1DIUk",
  "authDomain": "dashboard-50078.firebaseapp.com",
  "databaseURL": "https://dashboard-50078-default-rtdb.firebaseio.com",
  "storageBucket": "dashboard-50078.appspot.com"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)

# Get a reference to the database service
db = firebase.database()
user='shah'
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/diet', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        disease = request.form.get('disease')
        calories = request.form.get('calories')
        location=request.form.get('location')


        # Generate the diet plan based on the received form data (replace with your actual logic)
        diet_plan = generate_diet_plan(age, gender, disease, calories,location)

        # Render the HTML template with the diet plan
        return render_template('index.html', diet_plan=diet_plan)
    else:
        return render_template('index.html')

def generate_diet_plan(age, gender, disease, calories,location):
# Get the list of values under physical act data/kaushal
    physical = db.child("physActData").child(user).get().val()
    sleep = db.child("sleepData").child(user).get().val()
    mood = db.child("moodData").child(user).get().val()
    social = db.child("socIntData").child(user).get().val()
    stress = db.child("stressData").child(user).get().val()
    heart = db.child("heartRateData").child(user).get().val()

    # Get the value of the last element of the list
    physical = str(physical[-1])
    sleep = str(sleep[-1])
    mood = str(mood[-1])
    social = str(social[-1])
    stress = str(stress[-1])
    heart = str(heart[-1])

    age=str(age)
    calories=str(calories)
    message="sleep : " + sleep + ", heart rate : " + heart + ", mood : "+ mood+ ", physical activity level : " + physical+' calories : '+calories+', social: '+social +  ", stress :" +stress+ ", age: "+ age+ ", gender: "+gender+ ', disease: '+disease+', location : '+location+", prepare a diet plan for me based on these mentioned details , give answer as if you are a doctor and a dietician , prepare diet plan along with calories and all like calories intake ,energy intake ,vitamin intake for breakfast,lunch dinner and snacks etc, also mention the total calorie,vitamins,energy,protien intake at braekfast,lunch and dinner, intotal ,at the end tell a quote regarding mental fitness ,  and a joke to relieve my stress and then tell the user to shift to a vegetarian diet with reasons. NOTE : ANSWER LIKE A DOCTOR AND A DIETICIAN "

    if message:
        try:
            if len(message.split()) > 300:
                raise ValueError("Input contains more than 300 words. Please try again.")
            chat = openai.Completion.create(engine="text-davinci-003",prompt=message,max_tokens=3896,temperature=0.2)

        except ValueError as e:
            print(f"Error: {e}")
    reply = chat.choices[0].text
    response_message=f"{reply}"
    return response_message

if __name__ == '__main__':
    app.run(debug=True)



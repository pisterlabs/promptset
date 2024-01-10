from flask import Flask, render_template, request, jsonify
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import openai
from questions import questions  # import the questions dictionary

# Set up your OpenAI API credentials
openai.api_key = 'your-key-here' 
app = Flask(__name__)

current_state = "offer_type"
previous_answers = {}

@app.route('/')
def index():
    return render_template('index.html')

def get_matching_score(user_input, choices):
    result = process.extractOne(user_input, choices)
    return result

# List of question states for which you want to check for a match
match_check_states = ["offer_type", "property_type"]
match_check_states2 = ["terrain_status", "property_status"]


@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input, previous answers, and current state from the request
    user_input = request.json.get('user_input', '')
    previous_answers = request.json.get('previous_answers', {})
    current_state = request.json.get('current_state', 'offer_type')

    previous_answers[current_state] = user_input  # Save user's response

    # Check if user input matches any available responses for the current state
    if current_state in match_check_states:
        match = get_matching_score(user_input, questions[current_state]['responses'])
        if match:
            matched_response, score = match
            if score < 70:  # If the score is below the threshold
                return jsonify({
                    'question': "Sorry, I didn't understand that. Could you try again?",
                    'previous_answers': previous_answers,
                    'next_state': current_state
                })
            else:
                user_input = matched_response  # Use the matched response instead of the original user input

    # Check if user input matches any available responses for the current state for terrain_status and property_status 
    if current_state in match_check_states2:
        # If there are no responses available or user input is empty, skip matching
        if not questions[current_state]['responses'] or user_input.strip() == "":
            pass
        else:
            match = get_matching_score(user_input, questions[current_state]['responses'])
            if match:
                matched_response, score = match
                if score < 70:  # If the score is below the threshold
                    return jsonify({
                        'question': "Sorry, I didn't understand that. Could you try again?",
                        'previous_answers': previous_answers,
                        'next_state': current_state
                    })
                else:
                    user_input = matched_response  # Use the matched response instead of the original user input


    # Determine the next state based on the current state and user input
    if current_state == "property_type": 
        if user_input == "maison" or user_input == "appartement":
            current_state = "city"
        
        elif user_input == "terrain":
            current_state = "terrain_status"
        elif user_input in ["parking", "local commercial", "fond de commerce", "bastide", "château", "maison de village", "villa", "hôtel particulier", "ferme", "mas", "propriété", "rez de villa", "chalet", "duplex", "immeuble résidentiel", "loft", "rez de jardin", "studio", "triplex", "terrain constructible", "box de stockage", "cabanon", "cave", "garage", "viager", "bureaux", "immeuble commercial", "entrepôt"]:
            current_state = "Autre_Ville"
            
    elif current_state == "terrain_status":

        if user_input == "lotissement":
            current_state = "terrain_type"            
        elif user_input == "copropriété":  # If user_input is "copropriété"
            current_state = "number_of_lots"
        else :# If user_input is empty 
            current_state = "terrain_type"


    elif current_state == "property_status":
        if user_input == "lotissement":
            current_state = "property_style"            
        elif user_input == "copropriété":  # If user_input is "copropriété"
            current_state = "number_of_lots_AppartementMaison"
        else : 
            current_state = "property_style" # if user_input is empty 

    elif current_state in ['sale_price', 'rent_price']:
        current_state = 'charge'

    elif current_state == "charge":
        current_state = "Tonalite_de_l_annonce"

    elif current_state == "Tonalite_de_l_annonce":
        current_state = "Longueur_de_lannonce"
        
    elif current_state == "Longueur_de_lannonce":
        ad_prompt = previous_answers
        ad_message = create_advertisement(ad_prompt)
        return jsonify({'question': "Voici votre annonce généré par l'intelligence artificielle: " + ad_message, 'previous_answers': previous_answers})

        
    else:
        current_state = questions[current_state]['next']

    if current_state is None:
        # Check the offer type and decide on the next question
        offer_type = previous_answers.get('offer_type', '')
        if offer_type in ['vente', 'vente de programme neuf en état futur d\'achèvement', 'vefa','VEFA']:
            current_state = 'sale_price'
        elif offer_type in ['location', 'location saisonnière']:
            current_state = 'rent_price'
        
    return jsonify({'question': questions[current_state]['question'], 
                    'previous_answers': previous_answers, 
                    'next_state': current_state})

def create_advertisement(ad_prompt):
    print("--------------------------ad prompt:---------------------------")
    print("ad_prompt",ad_prompt)
    # Create a string by joining the non-empty values from the prompt
    # By iterating over the items in the ad_prompt dictionary, we maintain the order of the responses as they are received
    print("---------------------------ad prompt string---------------------")
    ad_prompt_string = ', '.join([f"{key.capitalize()}: {value}" for key, value in ad_prompt.items() if value.strip()])
    print("\n","ad_prompt_string ===>",ad_prompt_string,"\n")
    try:
        # Generate an advertisement using the OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                #{"role": "system", "content": "You are a real estate agent creating an advertisement for a property. Generate the ad in French. Write a compelling description using only the characteristics in the prompt."},
                {"role": "system", "content": "You are a real estate agent creating an advertisement for a property , generate the ad in French. Write a compelling description using only the characteristiques in the prompt that highlights the unique features and selling points of the property."},
                {"role": "user", "content": ad_prompt_string}
            ]

           
        )

    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please wait before making more requests.")
        return "Rate limit exceeded. Please wait before making more requests."

    message = response.choices[0].message['content'].strip()
    return message

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

# this is a chatbot to talk about metabolic health

# libraries
import openai # for AI
import streamlit as st # visual interface
import os
import requests

def format_for_llm(spot):
    name = spot.get('name', 'N/A')
    address = spot.get('vicinity', 'N/A')
    poi = spot.get('poi', 'N/A')  # Place of Interest (type or keyword)
    description = format_description(spot.get('types', []))

    return (f"📍 **{name}**\n"
            f"Located at: *{address}*\n"
            f"Category: *{poi}*\n"
            f"Type: *{description}*\n\n")

def format_description(types_list):
    # A mapping to improve the description
    improved_types = {
        'gym': 'Fitness Center',
        'health': 'Health Facility',
        # Add other improvements as needed
    }

    # Filter out generic types
    filtered_types = [t for t in types_list if t not in ['point_of_interest', 'establishment']]

    # Use our improved types mapping
    readable_types = [improved_types.get(t, t) for t in filtered_types]

    # Capitalize and replace underscores
    formatted_types = [t.replace('_', ' ').title() for t in readable_types]

    return ', '.join(formatted_types)

def search_exercise_spots_v4(zip_code, radius=1000, limit=3):

    places_interest = [
    {'type': 'park'},
    {'keyword': 'recreation center'},
    {'type': 'gym'},
    {'keyword': 'hiking trail'},
    {'keyword': 'public swimming pool'},
    {'keyword': 'cycling path'},
    {'keyword': 'athletic field'},
    {'keyword': 'dance studio'},
    {'keyword': 'community garden'},
    {'type': 'beach'},
    {'type': 'stadium', 'keyword': 'tennis court'},
    {'type': 'stadium', 'keyword': 'basketball court'},
    {'keyword': 'skating rink'},
    {'type': 'gym', 'keyword': 'martial arts'},
    {'keyword': 'rowing club'},
    {'keyword': 'public stairs'}
]

    def get_lat_lng_from_zip(zip_code):
        """Get latitude and longitude from a given ZIP code."""
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
        'address': zip_code,
        'key': os.environ.get('GOOGLE_API_KEY', None),
    }
        response = requests.get(geocode_url, params=params)
        if response.status_code != 200:
            raise Exception("ERROR: Geocoding ZIP code failed.")
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            raise Exception(f"ERROR: {data['status']}")


    """Search for nearby public spots to exercise given a ZIP code, places of interest, and radius."""
    latitude, longitude = get_lat_lng_from_zip(zip_code)
    places_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
    results = []

    for place in places_interest:
        params = {
            'location': f"{latitude},{longitude}",
            'radius': radius,
            'key': os.environ.get('GOOGLE_API_KEY', None),
        }

        # Merge our default params with the current place of interest
        params.update(place)

        response = requests.get(places_url, params=params)
        if response.status_code != 200:
            raise Exception(f"ERROR: Searching for {place} failed.")
        data = response.json()
        if data['status'] == 'OK':
            # Limit the results based on the limit parameter (default to 3)
            limited_results = data['results'][:limit]
            
            # Add the place of interest and description for context
            for result in limited_results:
                result['poi'] = place.get('type') or place.get('keyword')
                result['description'] = ", ".join(result.get('types', []))
            
            results.extend(limited_results)

    formatted_data = ""
    for spot in results:
            formatted_data += format_for_llm(spot)

    return formatted_data

# acttivates openAI with the user prompt + predefined expertise as a metabolic health expert
def generate_response(myprompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Your role is to act like a nutrition and health expert. Don't diagnose. Use only peer-reviewed information for your responses. At the end, provide resources they can read on their own. ###" + myprompt,
        temperature=.02,
        max_tokens=1024
    )
    return response.choices[0].text.strip()

# gathers user input
def main ():
    st.title("Weekly nutrition and activity plan to improve cardiovascular health") # title in the visual interface
    sex_options = ["Female", "Male"]
    diet_options = ["Vegetarian", "Carnivorous"]
    plan_options = ["Diet", "Activity", "Diet & Activity"]
    # form to be submitted
    with st.form("Basic information required"):
        st.write("Basic information required")
        age = st.slider("Approximate age")
        bmi = st.slider("Approximate BMI")
        wt_in_lbs = st.slider("Approximate weight in lbs", min_value=0, max_value=1000)
        # select sex
        sex = st.selectbox('Choose sex at birth:', sex_options)
        # dietary preference
        diet_pref = st.selectbox('Choose dietary preference:', diet_options)
        # plan preference
        plan_pref = st.selectbox('Choose plan preference:', plan_options)
        user_zip = st.text_input('Your zip code (optional):', key = 'Your zip code (optional):')
        health_issues = st.text_input('Exisiting medical issues (optional):', key = 'Exisiting medical issues (optional):')
        checkbox_val = st.checkbox("Confirm above information") 
        submitted = st.form_submit_button("Submit")

    if submitted:
       # join the above information in a single variable to be sent to AI
       myprompt = "I am %d years old %s, my weight is %d lbs and my BMI is %d. My dietary preference in %s. I want to improve my lifestyle and cardiovascular health. Make a weekly %s plan for me." % (age, sex, wt_in_lbs, bmi, diet_pref, plan_pref)
       if health_issues:
           myprompt = "I am %d years old %s, my weight is %d lbs and my BMI is %d. My dietary preference in %s. I want to improve my lifestyle and cardiovascular health. Make a weekly %s plan for me. While creating the plan, keep the following %s health issues in mind." % (age, sex, wt_in_lbs, bmi, diet_pref, plan_pref, health_issues)
       if user_zip:
           exercise_spots_v4 = search_exercise_spots_v4(user_zip, radius=1000)
           myprompt = "I am %d years old %s, my weight is %d lbs and my BMI is %d. My dietary preference in %s. I want to improve my lifestyle and cardiovascular health. Based on the above information, make a weekly %s plan for me. While creating the plan, keep the following %s health issues in mind. For the suggested physical activity in the plan, recommend an activity from the following %s" % (age, sex, wt_in_lbs, bmi, diet_pref, plan_pref, health_issues, exercise_spots_v4)
           
       st.write(generate_response(myprompt))


if __name__ == "__main__":
    main()
import streamlit as st
import openai
import googlemaps

openai.api_key = st.secrets["chatkey"]
gmaps = googlemaps.Client(key= st.secrets["googlekey"])

st.markdown("""
    <style>
        /* Main background color */
        body {
            background-color: #f0f0f0;
        }

        /* Set textbox border color and focus color */
        div[data-baseweb="input"] > div {
            border-color: teal !important;
        }
        div[data-baseweb="input"]:focus-within > div {
            border-color: teal !important;
            box-shadow: 0 1px 1px rgba(0,0,0,0.075), 0 0 0 0.2rem rgba(0,128,128,0.25) !important;
        }

        /* Adjust button colors and hover effect */
        .stButton>button {
            background-color: teal !important;
            border-color: teal !important;
            color: white !important;
        }
        .stButton:hover>button {
            background-color: #009688 !important;  /* Slightly darker teal for hover */
        }

    </style>
    """, unsafe_allow_html=True)

def get_recommendations(user_input):
    
    # Update status text for OpenAI query
    status_text.text("Progress: 10% - Processing your requirements...")
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
            {"role": "system", "content": "Create searchable text based on the users input to send to Google Maps. You are a local tour guide in the city of Chicago helping remote workers find somewhere to temporarily work from for a day like a coffee shop. The output should not contain any quotes."},
            {"role": "user", "content": user_input}
        ]
    )
    recommendations = response.choices[0].message.content
    query = recommendations

    # Update status text for Google Maps query
    status_text.text("Progress: 40% - Searching Google Maps...")
    progress_bar.progress(40)
    result = gmaps.places(query=query, location="Chicago", type='cafe')
    places = result.get('results', [])
    
    # Update status text for filtering places
    status_text.text("Progress: 60% - Filtering places based on ratings and status...")
    progress_bar.progress(60)
    filtered_places = [place for place in places if place.get('rating', 0) > 3.5 and place.get('business_status') == "OPERATIONAL"]
    
    # Sort the places by rating in descending order
    sorted_places = sorted(filtered_places, key=lambda x: x['rating'], reverse=True)
    
    # Limit the results to the top 7 places
    top_7_places = sorted_places[:7]
    
    # Extract names, addresses, and ratings
    names = [place.get('name', 'N/A') for place in top_7_places]
    addresses = [place.get('formatted_address', 'N/A') for place in top_7_places]
    ratings = [place.get('rating', 0) for place in top_7_places]
    
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
            {"role": "system", "content": "You are a local tour guide for the city of Chicago. You will review the users input of places they are considering working from and their requirements and explain why the places are a good fit for them. List out the name and give a 3-5 sentence answer as to why it matches their criteria."},
            {"role": "user", "content": "I wanted to look for: " + query + ". Here are my thoughts:" + ', '.join(names)}
        ]
    )
    
    recommendations = response.choices[0].message.content
    entries = recommendations.split("\n\n")
    reasoning = [entry.split(": ")[1] for entry in entries]

    return names, addresses, ratings, reasoning

# Fun title
st.markdown("<h1 style='text-align: center; color: teal;'>Where do you want to work from today? 🌆</h1>", unsafe_allow_html=True)

# Fun description and note about including a neighborhood
st.markdown("Enter your requirements and let's find the perfect coffee shop for you to work from in Chicago! ☕🏙️", unsafe_allow_html=True)
st.markdown("**Note**: Please include a neighborhood in your query for best results. 🌍", unsafe_allow_html=True)

# Text input for user's requirements
user_input = st.text_input("Tell us your requirements:")

# Example input
st.markdown('Example: _"Find me coffee shops in Lakeview that have an outdoor patio"_ or _"Wicker Park coffee shops that are pet friendly and have food."_', unsafe_allow_html=True)


if st.button("Submit"):
    if user_input:
        # Initialize the progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Progress: 0% - Starting the search...")
        
        # Fetch and display the recommendations
        names, addresses, ratings, reasoning = get_recommendations(user_input)
        
        # Update status for finalizing recommendations
        status_text.text("Progress: 80% - Finalizing recommendations...")
        progress_bar.progress(80)

        for n, a, r, reason in zip(names, addresses, ratings, reasoning):
            st.markdown(f"### 🏢 **{n}**")
            st.markdown(f"📍 [**{a}**](https://www.google.com/maps/search/?api=1&query={a.replace(' ', '+')})")
            st.markdown(f"⭐ Google Rating: **{r}**")
            st.markdown(f"🔍 **Here's the tea**: {reason}")
            st.write("---")

        # Completing the progress bar
        progress_bar.progress(100)
        status_text.text("Done! 🎉 Here are the best spots for you!")

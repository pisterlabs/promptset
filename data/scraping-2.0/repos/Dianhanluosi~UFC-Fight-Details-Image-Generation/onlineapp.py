import streamlit as st
import requests
import openai

# alle api key, empty
openai.api_key = ''

st.set_page_config(page_title="UFC Fights", layout="wide")
st.title('UFC Fights Details and Generated Image')

#sportsradar api key
api_key = 'eb3512e7350b4608a58b645a74690aa8'
schedule_url = 'https://api.sportsdata.io/v3/mma/scores/json/Schedule/ufc/'
event_url = 'https://api.sportsdata.io/v3/mma/scores/json/Event/{eventid}?key=' + api_key

#function, get fight schedule
def get_fight_schedule(season):
    url = f"{schedule_url}{season}?key={api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

#function, get specific event
def get_event_details(event_id):
    url = event_url.format(eventid=event_id)
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

#function, generate image based on event
def generate_image(prompt):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        return response.data[0].url
    except Exception as e:
        st.error(f'Failed to generate image: {e}')
        return None

#sidebar, scrolldown inputs
with st.sidebar:
    st.header("Settings")
    season = st.text_input('Enter the year:', value='2023')

    # input for api key
    openai.api_key = st.text_input('Enter OpenAI API Key:', type='password')

    if season:
        schedule = get_fight_schedule(season)
        if schedule:
            fight_names = [fight['Name'] for fight in schedule]
            selected_fight = st.selectbox('Select a Fight:', options=fight_names[::-1])

            selected_fight_id = {fight['Name']: fight['EventId'] for fight in schedule}[selected_fight]
            event_details = get_event_details(selected_fight_id)
            if event_details:
                fight_options = ["{} vs {}".format(fight['Fighters'][0]['FirstName'] + ' ' + fight['Fighters'][0]['LastName'],
                                                   fight['Fighters'][1]['FirstName'] + ' ' + fight['Fighters'][1]['LastName'])
                                 for fight in event_details['Fights']]

                selected_fight_detail = st.selectbox('Select a Fight Detail:', options=fight_options)

#display fight details + generated image
if 'selected_fight_detail' in locals() :
    st.subheader("Fight Details")
    selected_fight_info = next((fight for fight in event_details['Fights'] if "{} vs {}".format(fight['Fighters'][0]['FirstName'] + ' ' + fight['Fighters'][0]['LastName'], fight['Fighters'][1]['FirstName'] + ' ' + fight['Fighters'][1]['LastName']) == selected_fight_detail), None)
    
    if selected_fight_info:
        for fighter in selected_fight_info['Fighters']:
            fighter_name = f"{fighter['FirstName']} {fighter['LastName']}"
            moneyline = fighter['Moneyline']
            
            if moneyline > 0:
                moneyline_label = f"Underdog (+{moneyline})"
            elif moneyline < 0:
                moneyline_label = f"Favored ({moneyline})"
            else:
                moneyline_label = "Moneyline information not available"

            record = f"{fighter['PreFightWins']}-{fighter['PreFightLosses']} (Wins-Losses)"
            
            st.write(f"{fighter_name}: {record}, {moneyline_label}")

        if selected_fight_info['Fighters'][0]['Winner']:
            winner = f"{selected_fight_info['Fighters'][0]['FirstName']} {selected_fight_info['Fighters'][0]['LastName']}"
        elif selected_fight_info['Fighters'][1]['Winner']:
            winner = f"{selected_fight_info['Fighters'][1]['FirstName']} {selected_fight_info['Fighters'][1]['LastName']}"
        else:
            winner = "Winner information not available. This fight likely hasn't taken place yet. Or that it was canceled."
        st.write(f"Winner: {winner}")

    #display generated image
    if " vs " in selected_fight_detail and openai.api_key:
        fighter1, fighter2 = selected_fight_detail.split(" vs ")
        fight_prompt = f"UFC fighter {fighter1} fighting UFC fighter {fighter2} in {selected_fight} event"
        fight_image_url = generate_image(fight_prompt)
        if fight_image_url:
            st.image(fight_image_url)

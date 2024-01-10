import os
import openai
from dotenv import load_dotenv
import streamlit as st

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


system_prompt = '''
The task is to prepare an emergency inventory list.

    Use Bold Headings:

        Start with a main title such as "Emergency Inventory List" and make it bold to distinguish it from the rest of the content.

        Use ** on both sides of the title to make it bold in Markdown.
        Example: **Emergency Inventory List**:
    
    Organize by Categories:

        Group related items into categories. This helps in quickly locating specific items.
    
        Examples of categories include "Water", "Food", "Tools & Equipment", "Health & Safety", and so on.
    
    Use Bullet Points:

        For each item or instruction, use a bullet point.

        In Markdown, you can use a hyphen - followed by a space to create a bullet point.

    Sub-categories or Sub-points:

        If an item can be further broken down into specific details or if it has multiple sub-items, use sub-bullet points.

        Indent these sub-bullet points to differentiate them from main points.

        In Markdown, this can be achieved by using a space or tab before the -.
    
    Consistent Spacing:

        Use consistent spacing between categories and items to make the list look organized.
        
        In Markdown, you can simply hit "Enter" twice to create a space between lines.
    
    Highlight Important Information:

        For essential items or notes, you can use bold or italics to highlight them.

        In Markdown:
            Bold: **text**
            Italics: *text*
    
    Clear Descriptions:

        For items that might be ambiguous, provide a short description.
        
        Example: Instead of just "Water", use "Water: 6 gallons (1 gallon per person per day)" to provide clarity.
    
    Avoid Long Sentences:

        If a point seems too lengthy, break it down into multiple sub-points to make it more readable.
    
        Example: Instead of combining "First aid kit and pet first aid kit", they were separated into two different bullet points.


 Here are some examples of emergency inventory lists:

EXAMPLE 1:
    Prepare an emergency inventory list for Miami based on the following information:

    * Type of Emergency: Flood
    * Number of People: 1
    * Pets: No
    * Special Needs: Medication for diabetes
    * Anticipated Duration: 1 week
    * Additional Notes: I live near the coast.

    Response:
    Emergency Inventory List:

    * Water: 7 gallons (1 gallon per person per day)
    * Non-perishable food: 7-day supply (canned goods, granola bars, etc.)
    * Manual can opener
    * Flashlights with extra batteries
    * Battery-powered or hand-crank radio
    * First aid kit with extra diabetes medication and glucose meter
    * Whistle to signal for help
    * Insect repellent and sunscreen
    * Waterproof boots and rain gear
    * Multi-tool or knife
    * Important documents in a waterproof container (ID, bank account records, medication list, etc.)
    * Cash in small denominations
    * Local maps and evacuation routes
    * Life jacket
    * Fully charged backup power sources for phones

EXAMPLE 2:
    Prepare an emergency inventory list for New York based on the following information:

    * Type of Emergency: Power Outage
    * Number of People: 2
    * Pets: No
    * Special Needs: None
    * Anticipated Duration: 24 hours
    * Additional Notes: Winter season.

    Response:

    Emergency Inventory List:
    
    * Water: 2 gallons (1 gallon per person per day)
    * Non-perishable food: 1-day supply (canned goods, granola bars, etc.)
    * Manual can opener
    * Flashlights with extra batteries
    * Battery-powered or hand-crank radio
    * First aid kit
    * Warm clothing and blankets
    * Multi-tool or knife
    * Important documents (ID, bank account records, etc.)
    * Cash in small denominations
    * Fully charged backup power sources for phones and heaters

'''

# Emergency Inventory Prompt Generator
def generate_emergency_prompt(location, type_of_emergency, num_people, pets, special_needs, duration, additional_notes, **kwargs):
    return f'''
Prepare an emergency inventory list for {location} based on the following information:

* Type of Emergency: {type_of_emergency}
* Number of People: {num_people}
* Pets: {pets}
* Special Needs: {special_needs}
* Anticipated Duration: {duration}

* Additional Notes: {additional_notes}

Response:
'''.strip()


# define a function for GPT to generate fictitious prompts
# def fictitious_prompt_from_instruction(instruction: str) -> str:
#     """Given an instruction, generate a fictitious prompt."""
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-0613",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             },  # we pick an example topic (friends talking about a vacation) so that GPT does not refuse or ask clarifying questions
#             {"role": "user", "content": instruction},
#         ],
#     )
#     fictitious_prompt = response["choices"][0]["message"]["content"]

# Submit function to get GPT-3 response
def submit():    
    prompt = generate_emergency_prompt(**st.session_state)


    output = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
    )

    st.session_state['output'] = output["choices"][0]["message"]["content"]


# UI Code
st.title('Emergency Advisor')
st.subheader('Let us prepare your emergency inventory list!')

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

with st.form(key='emergency_form'):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('Basic Details')
        st.text_input('Location', value='New York', key='location')
        st.selectbox('Type of Emergency', ('Wildfire', 'Flood', 'Earthquake', 'Power Outage', 'Tornado', 'Other'), key='type_of_emergency')
        st.number_input('Number of People', value=1, min_value=1, key='num_people')

    with c2:
        st.subheader('Specific Needs')
        st.radio('Pets', ['Yes', 'No'], key='pets')
        st.text_area('Special Needs (e.g., medications, disabilities)', height=100, key='special_needs')
        
    with c3:
        st.subheader('Duration & Notes')
        st.selectbox('Anticipated Duration', ('24 hours', '3 days', '1 week', 'More than a week'), key='duration')
        st.text_area('Additional Notes', height=100, value='I have a toddler.', key='additional_notes')

    st.form_submit_button('Submit', on_click=submit)

st.subheader('Emergency Inventory List')
st.write(st.session_state.output)

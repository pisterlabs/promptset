import random
import openai
import pandas as pd
import streamlit as st
from hatch_prompts import generate_hatch_list, generate_pattern_materials_list

from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")

# Headers
st.header("Match the Hatch")
st.write("Enter some details about your upcoming fly fishing trip to get a list of likely aquatic insect hatches you will encounter, and related flies you should take.")
st.write("You can also generate a list of materials you will need to tie some suggested imitations.")
st.write("Powered by Pulze AI")

loading_messages = [
    "Checking under rocks...",
    "Asking the local guides...",
    "Consulting the fishing gods..."
]

# Initial Load
if 'trips' not in st.session_state:
    st.session_state["display_form"] = True
    st.session_state['trips'] = {
        "Wyoming-Green River-Cutthroat Trout-Early July": {
            "state": "Wyoming",
            "body_of_water": "Green River",
            "target_species": "Cutthroat Trout",
            "season": "Early July",
            "hatches": {'Blue Winged Olives': [{'pattern': 'Blue Winged Olive Dry', 'hook_size': '16-18', 'description': 'Gray or olive body with dark gray wings'}, {'pattern': 'Pheasant Tail Nymph', 'hook_size': '16-18', 'description': 'Brown or olive body with pheasant tail fibers and a copper bead head'}], 'Caddisflies': [{'pattern': 'Elk Hair Caddis', 'hook_size': '14-18', 'description': 'Light tan or brown body with elk hair wings and a brown hackle'}, {'pattern': 'Green Rockworm', 'hook_size': '14-18', 'description': 'Olive or green body with a dark head and sparse green or black hackle'}], 'Pale Morning Duns (Ephemerella infrequens)': [{'pattern': 'Pale Morning Dun Dry', 'hook_size': '14-16', 'description': 'Light yellow or cream body with light gray wings'}, {'pattern': 'RS2 Emerger', 'hook_size': '14-16', 'description': 'Gray body with a silver bead head and sparse gray wing buds'}], 'Midges': [{'pattern': "Griffith's Gnat", 'hook_size': '20-24', 'description': 'Black body with a white or gray wing and grizzly hackle'}, {'pattern': 'Zebra Midge', 'hook_size': '20-24', 'description': 'Black body with silver or red wire wrap and silver bead head'}], 'Stoneflies': [{'pattern': "Pat's Rubber Legs", 'hook_size': '6-10', 'description': 'Dark brown or black body with rubber legs and a touch of orange'}, {'pattern': "Kaufmann's Golden Stone", 'hook_size': '6-10', 'description': 'Yellow or tan body with brown markings and a gold bead head'}]},
            "materials": {}
        },
        "New York-Delaware River-Brown Trout-mid summer": {
            "state": "New York",
            "body_of_water": "Delaware River",
            "target_species": "Brown Trout",
            "season": "mid summer",
            "hatches": {'Sulphur Mayflies (Ephemerella invaria)': [{'pattern': 'Sulphur Dry Fly', 'hook_size': '14-16', 'description': 'Light yellow body with light gray wings and yellow hackle'}, {'pattern': 'Sulphur Emerger', 'hook_size': '14-16', 'description': 'Light yellow body with a split wing and yellow thorax'}], 'Caddisflies': [{'pattern': 'Elk Hair Caddis', 'hook_size': '12-16', 'description': 'Brown body with light brown wings and elk hair wings'}, {'pattern': 'Shop Vac', 'hook_size': '14-16', 'description': 'Olive or tan body with a silver bead and a sparse wing case'}], 'Blue Winged Olives (Baetis tricaudatus)': [{'pattern': 'Blue Winged Olive Dry Fly', 'hook_size': '18-22', 'description': 'Grayish olive body with gray wings and olive hackle'}, {'pattern': 'RS2 Emerger', 'hook_size': '18-22', 'description': 'Olive body with a small wing case and a thin profile'}], 'Tricos (Tricorythodes stygiatus)': [{'pattern': 'Trico Spinner', 'hook_size': '20-24', 'description': 'Black body with clear wings and a sparse hackle'}, {'pattern': 'Trico Dun', 'hook_size': '20-24', 'description': 'Gray body with clear wings and a thin profile'}], 'Pale Morning Duns (Ephemerella dorothea)': [{'pattern': 'PMD Comparadun', 'hook_size': '14-18', 'description': 'Pale yellow body with pale gray wings and a sparse hackle'}, {'pattern': 'PMD Sparkle Dun', 'hook_size': '14-18', 'description': 'Pale yellow body with pale gray wings and a light ribbing'}]},
            "materials": {}
        }
    }
    st.session_state["selected_trip"] = None
    st.session_state["api_key"] = None

def get_trip_key(state, body_of_water, target_species, season):
    return f"{state}-{body_of_water}-{target_species}-{season}"

def display_trip_recs(state, body_of_water, target_species, season):
    if not state or not body_of_water or not target_species or not season:
        st.error("Missing trip information", icon="🚨")
    else:
        trip_key = get_trip_key(state, body_of_water, target_species, season)
        if st.session_state['trips'].get(trip_key):
            st.session_state['selected_trip'] = trip_key
        else:
            with st.spinner(loading_messages[random.randint(0, len(loading_messages) - 1)]):
                hatches = generate_hatch_list(state, body_of_water, target_species, season, st.session_state["api_key"])
                st.session_state['trips'][trip_key] = {
                    "hatches": hatches,
                    "state": state,
                    "body_of_water": body_of_water,
                    "target_species": target_species,
                    "season": season,
                }
                st.session_state['selected_trip'] = trip_key
                st.success('Done!')

# Require Pulze API key on initial load
if not st.session_state["api_key"]:
    with st.form("api_key_submission"):
        st.write("Enter the following api keys. An OpenAI key is requested in addition to the Pulze key as a fallback")
        api_key = st.text_input('Pulze API Key', placeholder='Pulze API Key')
        open_ai_api_key = st.text_input('OpenAI API Key', placeholder='OpenAI API Key')
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state["api_key"] = api_key
            openai.api_key = open_ai_api_key
            st.rerun()
else:
    main_col1, main_col2 = st.columns(2)

    # Trip creation on left side of browser
    with main_col1:
        if not st.session_state["display_form"]:
            with st.form("display_trip_selector"):
                submitted = st.form_submit_button("New Trip")
                if submitted:
                    st.session_state["display_form"] = True
                    st.session_state["selected_trip"] = None
                    st.rerun()


        if st.session_state["display_form"]:
            with st.form("trip_selector"):
                st.write("Where are you headed?")
                col1, col2 = st.columns(2)

                with col1:
                    state = st.text_input('Location', placeholder='Germany', key="state")
                    body_of_water = st.text_input('Body of Water', placeholder='River Elbe', key="water")
                
                with col2:
                    target_species = st.text_input('Target Species', placeholder='Brown Trout', key="target_species")
                    season = st.text_input('Season', placeholder='Early July', key="season")

                submitted = st.form_submit_button("Generate predicted hatches")
                if submitted:
                    display_trip_recs(state, body_of_water, target_species, season)
                    st.session_state["display_form"] = False           
                    st.rerun()

    
        # Display previously generated trips
        data = []
        for trip_id, trip_info in st.session_state["trips"].items():
            data.append([trip_info["state"], trip_info["body_of_water"], trip_info["target_species"], trip_info["season"]])
        df = pd.DataFrame(data, columns=["Destination", "Water", "Species", "Season"])

        builder = GridOptionsBuilder.from_dataframe(df)
        builder.configure_selection(selection_mode='single', use_checkbox=False)
        grid_options = builder.build()
        st.write("Trips")
        return_value = AgGrid(df, gridOptions=grid_options)
        if return_value['selected_rows']:
            selected_trip = return_value['selected_rows'][0]
            trip_key = get_trip_key(selected_trip["Destination"], selected_trip["Water"], selected_trip["Species"], selected_trip["Season"])
            st.session_state["selected_trip"] = trip_key

        
    # Fly pattern display on right side of browser
    with main_col2:
        if st.session_state["selected_trip"]:
            selected_trip = st.session_state["trips"].get(st.session_state["selected_trip"])

            columns = ["Insect", "Pattern", "Hook Size", "Description"]
            hatch_rows = []
            for hatch, flies in selected_trip["hatches"].items():
                for fly_info in flies:
                    hatch_rows.append([hatch, fly_info["pattern"], fly_info["hook_size"], fly_info["description"]])
            st.write(f"Recommended flies for your trip to {selected_trip.get('body_of_water')} in {selected_trip.get('state')}")
            st.table(pd.DataFrame(hatch_rows, columns=columns))


            # Shopping list creation and download
            patterns_to_materials = selected_trip.get("materials", {})
            if not len(patterns_to_materials):
                with st.form("generate_materials_list"):
                    submitted = st.form_submit_button("Generate Materials List")
                    if submitted:
                        with st.spinner(loading_messages[random.randint(0, len(loading_messages) - 1)]):
                            patterns_to_materials = generate_pattern_materials_list(selected_trip.get("hatches"), api_key=st.session_state["api_key"])
                            st.session_state["trips"][st.session_state["selected_trip"]]["materials"] = patterns_to_materials
                            st.success('Done!')
                            st.rerun()
            
            if len(patterns_to_materials):
                materials_data = []
                for pattern, materials in patterns_to_materials.items():
                    for material in materials:
                        materials_data.append([pattern, material[0], material[1]])
                data = pd.DataFrame(materials_data, columns=["Pattern", "Type", "Material"])

                st.download_button(
                    "Download Shopping List",
                    data.to_csv().encode('utf-8'),
                    "materials_shopping_list.csv",
                    "text/csv",
                    key='download-csv'
                )
                            

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from funcs import initialize_google_sheet, write_to_google_sheet
import json

db = initialize_google_sheet(sheet_id='1I21v0eu5sAeEb0ZwirantxABsx6E8Cnr2DlZACEkgAY', sheet_name='Sheet5')


main_description_keys = ['protected_area_name', 'protected_area_location', 'protected_area_recreational_opportunities', 'protected_area_landscape_features', 'protected_area_visitor_infrastructure']
optional_metadata_keys = ["protected_area_flora", "protected_area_fauna", "protected_area_history", "protected_area_geology", 
                          "protected_area_address", "protected_area_contacts", "protected_area_fees", "protected_area_camping_options",
                          "protected_area_parking", "protected_area_campfire_rules", "protected_area_food_storage_rules", "protected_area_dog_rules"]
present_optional_keys = []
for entry in db:
    area_data = entry["Data"]
    for key in optional_metadata_keys:
        if key in area_data:
            present_optional_keys.append(key)
    #print(area_data)

    key_mapping = {
    "protected_area_flora": "Plant Life",
    "protected_area_fauna": "Wildlife",
    "protected_area_history": "History",
    "protected_area_geology": "Geology",
    "protected_area_address": "Address",
    "protected_area_contacts": "Contacts",
    "protected_area_fees": "Park Fees",
    "protected_area_camping_options": "Camping Options",
    "protected_area_parking": "Parking Information",
    "protected_area_campfire_rules": "Campfire Rules",
    "protected_area_food_storage_rules": "Food Storage Rules",
    "protected_area_dog_rules": "Pet Rules"
    }

    present_optional_keys = [key_mapping.get(key, key) for key in present_optional_keys]


    print(present_optional_keys)
    present_optional_keys = "; ".join(present_optional_keys)
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "RightOnTrek is a friendly encyclopedia for hikers in the US."},
        {"role": "user", "content": f"Write a RightOnTrek hiking encyclopedia article about an area using given data. Text structure: No heading; Main Description (size, location, geography, landscape and natural features, recreation, infrastructure); {present_optional_keys}. Character limit 2300 . Data: {area_data}"}
    ]
    )
    print(completion.choices[0].message)
    gpt4_text = completion.choices[0].message
    worksheet = write_to_google_sheet(SHEET_ID = '1I21v0eu5sAeEb0ZwirantxABsx6E8Cnr2DlZACEkgAY', SHEET_NAME = 'Sheet5')
    
    next_row = len(worksheet.col_values(3)) + 1

    # Append the forest_name and data_str to the sheet
    worksheet.update_cell(next_row + 3, 3, gpt4_text.content)
    print("Writing successful! Check Google Sheet.")
    break

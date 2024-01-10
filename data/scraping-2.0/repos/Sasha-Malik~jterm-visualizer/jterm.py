file = open("html_data", "r")

courses = file.read()
courses = courses.split("</li>")

from bs4 import BeautifulSoup

information = []
for html_string in courses : 

    soup = BeautifulSoup(html_string, 'html.parser')

    # Extract all the text from the parsed HTML
    all_text = soup.get_text(strip=True, separator='\n')

    information.append(all_text)

import folium
import re

# Sample data containing info strings as text
info_strings_text = information[:]

info_strings = []

# Function to parse an info string and extract information into a dictionary
def parse_info_string(info_text):
    info = {}
    lines = info_text.split('\n')
    info['name'] = lines[0]
    for line in lines[1:]:
        key_value = re.split(r':\s*', line)
        if len(key_value) == 2:
            key, value = key_value
            info[key.lower()] = value.strip()
    return info

# Iterating through the info string texts and parsing them into dictionaries
for info_text in info_strings_text:
    info_dict = parse_info_string(info_text)
    info_strings.append(info_dict)


import folium
import openai 

# api_key = ""  # Replace with an actual API key

# # Function to get latitude and longitude from the GPT API
# def get_lat_long_from_city_name(city_name):
#     prompt = f"""Get latitude and longitude of {city_name}, and provide the output in float as "latitude,longitude". """
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=50,
#         api_key=api_key
#     )
#     lat_long_text = response.choices[0].text.strip()
#     # lat, long = map(float, lat_long_text.split(","))
#     # return lat, long
#     return lat_long_text

info_strings.pop()
courses = info_strings
countries = ["United States", "Canada", "Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica", "Antigua and/or Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Bouvet Island", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Cook Islands", "Costa Rica", "Croatia (Hrvatska)", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "France, Metropolitan", "French Guiana", "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heard and Mc Donald Islands", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran (Islamic Republic of)", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, Democratic People's Republic of", "Korea, Republic of", "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libyan Arab Jamahiriya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Macedonia", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia, Federated States of", "Moldova, Republic of", "Monaco", "Mongolia", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "Netherlands Antilles", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Reunion", "Romania", "Russian Federation", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia South Sandwich Islands", "Spain", "Sri Lanka", "St. Helena", "St. Pierre and Miquelon", "Sudan", "Suriname", "Svalbard and Jan Mayen Islands", "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan", "Tajikistan", "Tanzania, United Republic of", "Thailand", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States minor outlying islands", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City State", "Venezuela", "Vietnam", "Virgin Islands (British)", "Virgin Islands (U.S.)", "Wallis and Futuna Islands", "Western Sahara", "Yemen", "Yugoslavia", "Zaire", "Zambia", "Zimbabwe"]
country_list = []
for c in countries:
    country_list.append(c.lower())


ad_trip = {}
for string in info_strings:
    if "Abu Dhabi with international trip" == string['location']:
        #convert the 's' dictionary into one big string
        s = str(string)
        #check every word in s with every word in countries
        for word in s.split():
            if word in country_list:
                ad_trip[string['name']] = word.capitalize()
                break


# hash= {}
# for course in courses:
#     try: 
#         city_name = course['location']
#         if city_name in hash:
#             continue
#         hash[city_name] = 1
#         print(city_name , get_lat_long_from_city_name(city_name))
#         # Specify the file name
#         file_name = "data.txt"

#         with open(file_name, "a") as text_file:
#             text_file.write(f"{city_name},{get_lat_long_from_city_name(city_name)};")
#     except:
#         continue


# Sample data containing course names and their corresponding locations
file_name = "data.txt"

try:
    with open(file_name, "r") as text_file:
        # Read the entire content of the file into a string
        file_content = text_file.read()

except FileNotFoundError:
    print(f"File '{file_name}' not found.")


city_info = {}
file_content = file_content.split(";")
for file in file_content:
    try:
        str = ''
        for f in file.split(",")[:-2]:
            str = str +","+ f 
        str = str[1:]
        name = str
        coordinates = [file.split(',')[-2],file.split(",")[-1]]
        city_info[f"{name}"] = coordinates
    except:
        continue

# creating a map 
map_center = [0, 0]
m = folium.Map(location=map_center, zoom_start=3)  

# adding markers to map
import random 

popups = {}
for latitude, longitude in city_info.values():
    latitude,longitude = float(latitude),float(longitude)
    popups[latitude, longitude] = []

m = folium.Map(location=[0, 0], zoom_start=3)

# Iterate through the courses and add popups to the popups dictionary
popups = {}

for course in courses:
    
        city_name = course['location']
        if city_name == "Abu Dhabi with international trip":
            city_name = "Abu Dhabi"

        if course['name'] in ad_trip:
            popup = [course['name'],course['term'],course['points'],ad_trip[course['name']]]

        else:
            popup = [course['name'],course['term'],course['points']]

        coords = city_info[city_name]
        latitude, longitude = float(coords[0]), float(coords[1])

        if (latitude, longitude) not in popups:
            popups[(latitude, longitude)] = []

        popups[(latitude, longitude)].append(popup)

    
# Iterating through the popups dictionary and adding markers to the map
for (latitude, longitude), popup_content in popups.items():
    
    popup_html = f'<div style="max-height: 200px; overflow-y: auto;">'
    
    # Adding each property to the popup
    for course_properties in popup_content:
        popup_html += f'<b>Name:</b> {course_properties[0]}<br>'
        popup_html += f'<b>Term:</b> {course_properties[1]}<br>'
        popup_html += f'<b>Points:</b> {course_properties[2]}<br>'
        
        # Checking if there is a fourth property before adding it
        if len(course_properties) == 4:
            # print(course_properties[3])
            popup_html += f'<b>International Trip:</b> {course_properties[3]}<br>'

        popup_html += '<br>'
    
    popup_html += '</div>'
    
    folium.Marker(
        location=[latitude, longitude],
        popup=popup_html,
        icon=folium.Icon(icon='cloud')
    ).add_to(m)

# Saving the map
m.save('course_locations_map.html')


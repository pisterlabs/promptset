import requests
import json
import streamlit as st
import openai
from langchain.tools import DuckDuckGoSearchRun
import string
import googlemaps
from datetime import datetime, timedelta
import os
import cohere

def assistant_code(city_name, from_date, to_date, hotel_name):
    api_key =  os.getenv('OPENWEATHER_API_KEY')
    co_api_key = os.getenv('CO_API_KEY')
    g_api_key =  os.getenv('GOOGLE_API_KEY')
    base_url =  os.getenv('OPENWEATHER_BASE_URL')
    co = cohere.Client(co_api_key)
    duration = (to_date-from_date).days
    temp = ''
    pressure = ''
    humidity = ''
    weather_description = ''
    if city_name is not None and hotel_name is not None and duration!=0:
        # Display user message in chat message container
        # st.chat_message("user").markdown(city_name)

        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)

        x = response.json()
        print(x)

        if(x['cod'] != '404'):
            y = x['main']
            temp = str(y['temp'])
            pressure = str(y['pressure'])
            humidity = str(y['humidity'])
            # st.markdown("Temperature: "+ temp)
            # st.markdown("Pressure: " + pressure)
            # st.markdown("Humidity: " + humidity)
            z = x["weather"]
    
            # store the value corresponding
            # to the "description" key at
            # the 0th index of z
            weather_description = z[0]["description"]
        
            # print following values
            # st.markdown("Description: " + weather_description)

        news_search = DuckDuckGoSearchRun(verbose=True)
        news = news_search.run(city_name+' social events, sports events and concerts, comedy club shows, restaurants, etc.')
        resto_search = DuckDuckGoSearchRun(verbose=True)
        resto = resto_search.run(city_name + " Popular affordable restaurants near me")
        print(resto_search)
        message = f'''
                    Create a travel itinerary for one day for the given city {city_name} for the given duration {duration} from {from_date} to {to_date}.
                    Include all the main attractions of the particular city based on knowledge. 
                    DO not write any code in python or any other language. Just write the body of the html page. (Do not write the title or the head), just write the body
                    As additional information include the news and restaurant options to be in the itinerary. Make sure some of these are in the itinerary. 
                    This is the news search: {news}
                    This is the restaurant search: {resto}
                    Give the travel itinerary for the appropriate weather conditions and the given news stories of current events in the city.
                    Include the following events in the itinerary as well: 
                    The weather conditions for the given city are:
                    Temperature (in Kelvin): {temp},
                    Pressure (in mm of Hg): {pressure},
                    Weather description: {weather_description}.
                    For each day, start the itinerary at 9:00AM and end it at 11:00PM.
                    If the itinerary is for two days, give the output as: 
                    Day 1:
                    9:00AM: Breakfast at <b>Restaurant 1</b>
                    .
                    .
                    .
                    11:00PM: Dinner at <b>Restaurant 2</b>
                    <br>
                    Day 2:
                    9:00AM: Breakfast at <b>Restaurant 3</b>
                    .
                    .
                    .
                    11:00PM: Dinner at <b>Restaurant 4</b>
                    Also make sure that the timings are properly coordinated. For example, if the event takes place between 6PM and 11PM, do not put that at 2PM on the itinerary, instead put it at the appropriate times.
                    Make the attractions in bold and the timings in italics.
                    DO NOT WRITE GENERIC PLACE NAMES LIKE:
                    Restaurant XYZ, Place Name 1, Area 1, etc. Be specific and give real names for the places by using the restaurant names given above
                    While writing the itinerary, use the location of the heading. For example: 
                    Always add the HTML bold tags before writing the place of attraction. After writing the place, remember to close the bold tag with </b> This is extremely important in the executions that follow. 
                    <b>place name<b> is not acceptable. <b>place name</b> is acceptable. These tags must be present in the final output.
                    Make sure you include the above formatting in the whole prompt. Do not use ### or ** or any other markdown syntax.
                    Give appropriate timings and give the itinerary and explain each attraction in great detail.
                    Avoid the introductions like "Sure..." or "Certainly...".
                    DO NOT WRITE ANYTHING LIKE "Would you like me to help you with anything else regarding travel planning?"
                    OR
                    "Toronto, Ontario, Canada, has a lot to offer and the weather conditions are just adding to the experience. The clear sky and sunny weather will provide the perfect backdrop for exploring the city."
                    OR ANYTHING OF THE SORT. Just give the itinerary and the attractions and the timings.
                    '''

        response = co.chat(message,
                        model='command',
                        temperature=0.0)

        answer = response.text
        answer = answer.replace("```", "")
        answer = answer.replace("** ", "</b> ")
        answer = answer.replace("**", "<b>")
        answer = answer.replace("###", "")
        answer = answer.replace("##", "")
        answer = answer.replace("\n", "<br>")
        lines_ans = answer.splitlines()
        print(lines_ans)
        attraction_list = []
        for i in range(len(lines_ans)):
            star_index = lines_ans[i].find("<b>")
            if(star_index != -1):
                attraction = ''
                for j in range(star_index+3, len(lines_ans[i])):
                    if(lines_ans[i][j] == "<"):
                        break
                    else:
                        attraction+=lines_ans[i][j]
                attraction_list.append(attraction)
        print("List: ", attraction_list)

        # st.markdown(answer)

            


        Maps = googlemaps.Client(key=g_api_key)

        waypoints = attraction_list
        print(len(waypoints))

        waypoints = [a + " " + city_name for a in waypoints]
        #waypoints.append('Your location')

        print(len(waypoints))
        #place = Maps.find_place(waypoints[0], input_type = "textquery")
        #print(place)
        id_list = []
        locat_id = Maps.find_place("Your location", input_type = "textquery")
        print(locat_id)
        # for i in waypoints:
        #     place_ids = Maps.find_place(i, input_type = "textquery")
        #     pl_id = place_ids['candidates'][0]['place_id']
        #     #print(pl_id)
        #     id_list.append(pl_id)
        

        results = Maps.directions(origin = hotel_name,
                                                destination = hotel_name,                                    
                                                waypoints = waypoints,
                                               optimize_waypoints = True
        )

        print(results)

        for i, leg in enumerate(results[0]["legs"]):
            print("Stop:" + str(i),
                leg["start_address"], 
                "==> ",
                leg["end_address"], 
                "distance: ",  
                leg["distance"]["value"], 
                "traveling Time: ",
                leg["duration"]["value"]
            )

        locations = attraction_list

        markers = ["color:blue|size:mid|label:" + chr(65+i) + "|" 
                        + r for i, r in enumerate(locations)]

        result_map = Maps.static_map(
                        center=hotel_name,
                        scale=2, 
                        zoom=12,
                        size=[640, 640], 
                        format="jpg", 
                        maptype="roadmap",
                        markers=markers,
                        path="color:0x0000ff|weight:2|" + "|".join(locations))

        with open("driving_route_map_notracks.jpg", "wb") as img:
            for chunk in result_map:
                img.write(chunk)

        marker_points = []
        waypoints = []

        #extract the location points from the previous directions function

        for leg in results[0]["legs"]:
            leg_start_loc = leg["start_location"]
            marker_points.append(f'{leg_start_loc["lat"]},{leg_start_loc["lng"]}')
            for step in leg["steps"]:
                end_loc = step["end_location"]
                waypoints.append(f'{end_loc["lat"]},{end_loc["lng"]}')
        last_stop = results[0]["legs"][-1]["end_location"]
        marker_points.append(f'{last_stop["lat"]},{last_stop["lng"]}')
                
        markers = [ "color:blue|size:mid|label:" + chr(65+i) + "|" 
                + r for i, r in enumerate(marker_points)]
        result_map = Maps.static_map(
                        center = hotel_name,
                        scale=2, 
                        zoom=13,
                        size=[640, 640], 
                        format="jpg", 
                        maptype="roadmap",
                        markers=markers, 
                        path="color:0x0000ff|weight:2|" + "|".join(waypoints))

        with open("driving_route_map.jpg", "wb") as img:
            for chunk in result_map:
                img.write(chunk)

        # st.image("driving_route_map.jpg")
        return answer, result_map

# if __name__ == "__main__":
#     app()

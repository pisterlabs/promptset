import json
import numpy as np
import src.utils.global_value as global_value
from src.agent.prompt_template import GeneralPromptTemplate, StreamPlan
from src.agent.recommendation import Recommender
from src.agent.route_plan import RoutePlanner
from src.tools.travel.google_map_client import google_map_client
from src.tools.travel.serpAPI import distance_calculator
from src.utils.debug import log_info, DEBUG
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

# Constants
PLAN_MODEL = "gpt-4-0613"
SPEED = 30000

class TravelAgent:
    """Class for handling the travel agent's operations."""

    def __init__(self, agent_id, departure, destination, days, additional_info=""):
        """Initializes the TravelAgent."""
        self.agent_id = agent_id
        self.departure = departure
        self.destination = destination
        self.days = days
        self.additional_info = additional_info
        self.whole_plans = []
        self.recommender = Recommender(self.destination, self.days, self.additional_info)
        self.gmaps = google_map_client(self.destination)

    def check_restaurants(self, current_time, restaurants_num, current_pos, points):
        """Checks and adds restaurants into the itinerary based on the current time and position."""
        if 12 < current_time < 18 and restaurants_num == 0:
            restaurant = self.gmaps.search_restaurants(current_pos)
            if restaurant:
                points.append(restaurant)
                restaurants_num += 1
                current_time += 1
        elif current_time > 18 and restaurants_num == 1:
            restaurant = self.gmaps.search_restaurants(current_pos)
            if restaurant:
                points.append(restaurant)
                restaurants_num += 1
                current_time += 1
        return current_time, restaurants_num, points

    def insert_restaurants(self, sights, initial_geo):
        """Inserts restaurants into the itinerary based on the route and sights."""
        points, current_time, current_pos, restaurants_num = [], 9, initial_geo, 0

        for sight in sights:
            points.append(sight)
            current_time += distance_calculator(current_pos, sight['gps_coordinates']) / SPEED
            current_pos = sight['gps_coordinates']
            current_time, restaurants_num, points = self.check_restaurants(current_time, restaurants_num, current_pos, points)
            current_time += sight['recommend_duration']
            current_time, restaurants_num, points = self.check_restaurants(current_time, restaurants_num, current_pos, points)
        
        return points

    def plan_one_day_itinerary(self, partition, day):
        """Plans a one-day itinerary based on a given set of destinations and the specific day."""
        
        with open("src/prompts/Travel/make_one_day_itinerary_prompt.txt") as f:
            travel_prompt = f.read()
        travel_prompt_template = GeneralPromptTemplate(
            template=travel_prompt,
            input_variables=["points", "distances"]
        )
        chat_llm = ChatOpenAI(
            temperature=0,
            model_name=PLAN_MODEL,
            streaming=True,
            callbacks=[StreamPlan(self.agent_id)]
        )
        llm_chain = LLMChain(
            llm=chat_llm,
            prompt=travel_prompt_template,
            verbose=DEBUG
        )
        agent = global_value.get_dict_value('agents', self.agent_id)

        if agent is None:
            return ""
        agent.UI_info.chatbots.append([None, f'Planning for day {day}...'])
        sights, hotel = partition
        points = [hotel]
        sights_restaurants = self.insert_restaurants(sights, hotel['gps_coordinates'])
        points.extend(sights_restaurants)
        points.append(hotel)
        log_info(f"Points are: {json.dumps(points, indent=4)}")
        distances = [
            f"From {point['name'] if 'name' in point else point['title']} to {points[i+1]['name'] if 'name' in points[i+1] else points[i+1]['title']}, "
            f"the traveling route is: {self.gmaps.directions(point['gps_coordinates'], points[i+1]['gps_coordinates'])}"
            for i, point in enumerate(points[:-1])
        ]
        agent.UI_info.travel_plans.append("")
        plan = llm_chain(
            {"points": json.dumps(points, indent=4), "distances": "\n".join(distances)},
            return_only_outputs=True
        )['text']

        return plan

    def plan_full_itinerary(self, remaining_sights, partitions):
        """Plans the full itinerary."""
        for day, partition in enumerate(partitions, start=1):
            daily_plan = self.plan_one_day_itinerary(partition, day)
            if not daily_plan:
                print("Failed to generate a plan for day", day) 
                return
            self.whole_plans.append(daily_plan)
            
        flight_info = self.gmaps.search_flights(self.departure, self.destination)
        flight_text = "Title: {}\nLink: {}\nSnippet: {}".format(
        flight_info["title"],
        flight_info["link"],
        flight_info["snippet"]
        )
        
        self.whole_plans.append("1. Flight Information:\n\n{}\n\n".format(flight_text))
        agent = global_value.get_dict_value('agents', self.agent_id)

        if agent is None:
            print("Agent not found") 
            return ""
        
        agent.UI_info.travel_plans.append(
            "This is the booking information.\n\n{}".format(self.whole_plans[-1])
        )



    def run(self):
        """Executes the travel agent's operations."""
        recommended_sights = self.recommender.search_famous_sights()
        sights_geo = np.array([(sight['gps_coordinates']['latitude'], sight['gps_coordinates']['longitude']) for sight in recommended_sights])
        hotel_geo = sights_geo.mean(axis=0)
        initial_hotel = self.gmaps.search_hotel(hotel_geo)

        route_planner = RoutePlanner(recommended_sights, initial_hotel)
        remaining_sights, partitions = route_planner.plan_full_route()

        self.plan_full_itinerary(remaining_sights, partitions)
from typing import List
from app.model import Message
import openai
import os
import json
from app.utils import logger, remove_spanish_special_characters
from datetime import datetime, timedelta, date


openai.api_key = os.environ.get('OPENAI_API_KEY')


def get_current_datetime():
    return datetime.now()

def calculate_date_based_on_day_name(date_from: datetime, day_name: str) -> datetime:
    """Useful for when you need to calculate a exact date in format YYYY-MM-DD given a day name i.e: Tuesday."""

    def _day_name_to_int(day_string) -> int:
        # Convert the day string to lowercase for case-insensitive comparison
        day_string_lower = remove_spanish_special_characters(day_string.lower())

        # Map day strings to day numbers
        day_mapping = {
            'lunes': 0,
            'martes': 1,
            'miercoles': 2,
            'jueves': 3,
            'viernes': 4,
            'sabado': 5,
            'domingo': 6,
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6
        }

        # Check if the day string exists in the mapping
        if day_string_lower in day_mapping:
            return day_mapping[day_string_lower]
        else:
            raise ValueError('Invalid day string')
    
    day_int = _day_name_to_int(day_name)
    # Get the current date
    current_date = date_from.date()

    # Get the next occurrence of the specified day name
    days_ahead = (7 + day_int - current_date.weekday()) % 7
    next_date = current_date + timedelta(days=days_ahead)

    return next_date

json_fn = {
    "name": "calculate_booking_info",
    "description": "Calculate the exact check-in and check-out dates for a reservation and the number of guests staying.",
    "parameters": {
        "type": "object",
        "properties": {
            "check_in_date": {
                "type": "string",
                "description": "If present in the conversation, the Check In date in the format: YYYY-MM-DD i.e: 2023-03-25"
            },
            "check_out_date": {
                "type": "string",
                "description": "If present in the conversation, the Check Out date in the format: YYYY-MM-DD i.e: 2023-03-25"
            },
            "check_in_dow": {
                "type": "string",
                "description": "If present in the conversation, the Check In day of the week.",
                "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            },
            "check_out_dow": {
                "type": "string",
                "description": "If present in the conversation, the Check Out day of the week.",
                "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            },
            "num_nights": {
                "type": "string",
                "description": "If present in the conversation, the number of nights the guests plan to stay"
            },
            "num_guests": {
                "type": "string",
                "description": "If present in the conversation, the number of guests staying"
            }
        },
        "required": []
    }
}

class SearchDataExtractor:


    def calculate_check_in_date(self, check_in_date, check_in_dow):
        if check_in_date is not None:
            if check_in_date > get_current_datetime().strftime("%Y-%m-%d"):
                return datetime.strptime(check_in_date, "%Y-%m-%d")
            else:
                logger.debug(f"check_in_date is from the passt: {check_in_date}")        
        if check_in_dow is not None:
            return calculate_date_based_on_day_name(get_current_datetime(), check_in_dow)
        logger.debug(f"""For some reason it is not possible to calculate the check_in date, 
                     check_in_date: {check_in_date}, 
                     check_in_dow: {check_in_dow}""")
        return None
    
    def calculate_check_out_date(self, check_in_date: str, check_out_date: str, check_out_dow: str, num_nights: int):
        if check_out_date is not None:
            if check_out_date > get_current_datetime().strftime("%Y-%m-%d") and check_out_date > check_in_date:
                return datetime.strptime(check_out_date, "%Y-%m-%d")
            else:
                logger.debug(f"""check_out_date is wrong. 
                             check_out_date: {check_out_date}, 
                             check_in_date: {check_in_date}""")  
        
        date_from = datetime.strptime(check_in_date, "%Y-%m-%d")

        if int(num_nights) > 0:
            return date_from + timedelta(days=num_nights)

        if check_out_dow is not None:
            return calculate_date_based_on_day_name(date_from, check_out_dow)
        
        logger.debug(f"""For some reason it is not possible to calculate the check_out date, 
                     check_in_date: {check_in_date}, 
                     check_out_date: {check_out_date}, 
                     check_out_dow: {check_out_dow},
                     num_nights: {num_nights}""")
        return None
    
    def get_num_days(self, start_date, end_date):
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)
        num_days = (end_date - start_date).days
        return num_days

    def calculate_booking_info(self,
                            fn_params: dict):
        check_in_date = fn_params.get("check_in_date", None)
        check_in_dow = fn_params.get("check_in_dow", None)

        check_out_date = fn_params.get("check_out_date", None)
        check_out_dow = fn_params.get("check_out_dow", None)
        num_nights = int(fn_params.get("num_nights", 0))

        num_guests = int(fn_params.get("num_guests", 0))

        if check_in_date is None and check_in_dow is None:
            logger.debug("Not possible to calculate check_in date")
            return None
        
        if check_out_date is None and check_out_dow is None and num_nights is None:
            logger.debug("Not possible to calculate check_out date")
            return None
        
        check_in_date = self.calculate_check_in_date(check_in_date, check_in_dow)

        if check_in_date is not None:
            check_in_date = check_in_date.strftime("%Y-%m-%d")
            check_out_date = self.calculate_check_out_date(check_in_date, check_out_date, check_out_dow, num_nights)
            if check_out_date is not None:
                check_out_date = check_out_date.strftime("%Y-%m-%d")

        num_nights = None
        if check_out_date is not None and check_in_date is not None:
            num_nights = self.get_num_days(check_in_date, check_out_date)

        return {
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "num_guests": num_guests,
            "num_nights": num_nights
        }

    def normalize_dow(self, params):
        map_days = {
            "lunes"
        }

    def run(self, messages: List[Message]):
        
        messages_input = [{"role": "system", "content": f"What are the exact check-in and check-out dates and number of guests for the reservation? IMPORTANT: Today is: {datetime.now().date().strftime('%Y-%m-%d')}"}]
        for msg in messages:
            messages_input.append({"role": msg.role, "content": msg.text})
        # messages_input.append("role")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages_input,
            functions=[json_fn],
            temperature=0., 
            max_tokens=500, 
        )

        if "function_call" in response.choices[0].message and "arguments" in response.choices[0].message["function_call"]:
            fn_parameters = json.loads(response.choices[0].message["function_call"]["arguments"])
            # fn_parameters["user_has_selected"] = ("bnbot_id" in fn_parameters and fn_parameters["bnbot_id"] != "")
            logger.debug(f"calculate_booking_info fn_parameters {fn_parameters}")

            return self.calculate_booking_info(fn_parameters)
        
        return None
        
        
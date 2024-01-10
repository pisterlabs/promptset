from pydantic import Field, BaseModel
from typing import List, Optional, Union
from langchain.agents import Tool
from langchain.agents.agent_toolkits.base import BaseToolkit
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any
from langchain.tools import BaseTool, tool
from restaurant_reservation_manager import RestaurantReservationManager
from ..calendar.google_calendar import GoogleCalendar
from ..knowledge_base.kb import knowledge_base
from ..common.utils import extract_phone_number
import os

google_calendar = GoogleCalendar(calendar_name="Restaurant Agent")
reservation_manager = RestaurantReservationManager(google_calendar)

SCRATCH_SPACE = os.getenv("SCRATCH_SPACE_DIR")

class ReservationsToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [
            find_tables_availability,
            find_ballroom_availability,
            finalize_table_reservation,
            finalize_ballroom_reservation,
            cancel_reservation,
            update_reservation_for_tables,
            update_reservation_for_ballrooms    
        ]

class CasualDiningReservationsToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [
            find_tables_availability,
            finalize_table_reservation,
            cancel_reservation,
            update_reservation_for_tables,
            send_menu
        ]   


class FindTablesAvailabilitySchema(BaseModel):
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    time: str = Field(description="The time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")

@tool("find_tables_availability", args_schema=FindTablesAvailabilitySchema)
def find_tables_availability(
    date: str,
    time: str
) -> Union[Dict[str, Any], str]:
    """ Use this to find availability for tables. Use this for party size of upto 6 people. """
    reservations = reservation_manager.find_tables_for_individuals(
        date = date, 
        time = time
    )
    return reservations

class FindBallroomAvailabilitySchema(BaseModel):
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    start_time: str = Field(description="The start time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")
    duration_in_hours: int = Field(description="The duration of the reservation in hours. Round it up.")
    
@tool("find_ballroom_availability", args_schema=FindBallroomAvailabilitySchema)
def find_ballroom_availability(
    date: str,
    start_time: str,
    duration_in_hours: int
) -> Union[Dict[str, Any], str]:
    """ Use whenever you want to find availability for ballrooms. Use this for party size of at least 25 people. """
    reservations = reservation_manager.find_ballrooms_availability(
        date = date,
        start_time = start_time,
        duration = duration_in_hours
    )
    return reservations

class FinalizeTableReservationSchema(BaseModel):
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    time: str = Field(description="The time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")
    party_size: int = Field(description="The size of the party")
    name: str = Field(description="The name of the person making the reservation")
    phone_number: str = Field(description="The phone number of the person making the reservation")

@tool("finalize_table_reservation", args_schema=FinalizeTableReservationSchema)
def finalize_table_reservation(
    date: str,
    time: str,
    party_size: int,
    name: str,
    phone_number: str
) -> Union[Dict[str, Any], str]:
    """ Use this to finalize a reservation for a table. """
    return reservation_manager.make_reservation_for_individuals(
        name = name,
        phone_number = phone_number,
        date = date,
        start_time = time,
        party_size = party_size,
    )

class FinalizeBallroomReservationSchema(BaseModel):
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    start_time: str = Field(description="The start time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")
    party_size: int = Field(description="The size of the party")
    duration_in_hours: int = Field(description="The duration of the reservation in hours. Round it up.")
    name: str = Field(description="The name of the person making the reservation")
    phone_number: str = Field(description="The phone number of the person making the reservation")

@tool("finalize_ballroom_reservation", args_schema=FinalizeBallroomReservationSchema)
def finalize_ballroom_reservation(
    date: str,
    start_time: str,
    name: str,
    phone_number: str,
    party_size: int,
    duration_in_hours: int
) -> Union[Dict[str, Any], str]:
    """ Use this to finalize a reservation for a ballroom. """
    return reservation_manager.make_reservation_for_ballrooms(
        name = name,
        phone_number = phone_number,
        date = date,
        start_time = start_time,
        party_size = party_size,
        duration_in_hours = duration_in_hours
    )

class CancelReservationSchema(BaseModel):
    reservation_id: str = Field(description="The id of the reservation to cancel")

@tool("cancel_reservation", args_schema=CancelReservationSchema)
def cancel_reservation(
    reservation_id: str
) -> str:
    """ Use this to cancel a reservation. """
    return reservation_manager.cancel_reservation(
        event_id = reservation_id
    )

class UpdateReservationForTablesSchema(BaseModel):
    event_id: str = Field(description="The id of the reservation to update")
    name: str = Field(description="The name of the person making the reservation")
    phone_number: str = Field(description="The phone number of the person making the reservation")
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    start_time: str = Field(description="The time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")
    party_size: int = Field(description="The size of the party")

@tool("update_reservation_for_tables", args_schema=UpdateReservationForTablesSchema)
def update_reservation_for_tables(
    event_id: str,
    name: str,
    phone_number: str,
    date: str,
    start_time: str,
    party_size: int,
) -> Union[Dict[str, Any], str]:
    """ Use this to update a reservation. """
    return reservation_manager.update_reservation(
        event_id = event_id,
        name = name,
        phone_number = phone_number,
        date = date,
        start_time = start_time,
        party_size = party_size
    )

class UpdateReservationForBallroomsSchema(BaseModel):
    event_id: str = Field(description="The id of the reservation to update")
    name: str = Field(description="The name of the person making the reservation")
    phone_number: str = Field(description="The phone number of the person making the reservation")
    date: str = Field(description="The date to find available reservations for. Format: YYYY-MM-DD")
    start_time: str = Field(description="The time for the reservation in 12-hour format. Always Format is as \"HH:MM AM/PM\"")
    party_size: int = Field(description="The size of the party")
    duration_in_hours: int = Field(description="The duration of the reservation in hours. Round it up.")

@tool("update_reservation_for_ballrooms", args_schema=UpdateReservationForBallroomsSchema)
def update_reservation_for_ballrooms(
    event_id: str,
    name: str,
    phone_number: str,
    date: str,
    start_time: str,
    party_size: int,
    duration_in_hours: int
) -> Union[Dict[str, Any], str]:
    """ Use this to update a reservation. """
    return reservation_manager.update_reservation(
        event_id = event_id,
        name = name,
        phone_number = phone_number,
        date = date,
        start_time = start_time,
        party_size = party_size,
        duration_in_hours = duration_in_hours
    )

class SendMenuSchema(BaseModel):
    phone_number: str = Field(description="The phone number of the customer")  


send_menu_file_path = f"{SCRATCH_SPACE}/send_menu.txt"
send_menu_file = open(send_menu_file_path, "a")

@tool("send_menu", args_schema=SendMenuSchema)
def send_menu(
    phone_number: str
) -> str:
    """ Use this to send the menu to a customer. """
    try:
        extracted_phone_number = extract_phone_number(phone_number)
        if len(extracted_phone_number) != 10:
            return "Invalid phone number. Please give me a valid phone number."
        send_menu_file.write(f"phone_number: {extracted_phone_number}\n")
        return "Menu sent."
    except Exception as e:
        return "I'm having trouble with the system. I will make a note and send it to you as soon as possible. Does that work?"
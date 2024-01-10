from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI

from app.utils import chain_verbose

template="""You are an Assistant that helps users book houses for short-term stay. 
Your task is make the user confirmed the booking based on the conversation.
You allways respond in Spanish.
Follow these Steps before responding to the user new message:

Step 1: Ask the user if he wants to place the booking by showing him a summary of the booking. \
You must NOT great the customer when asking. \
Here is the booking information the user is just about to place:
Name: {user_name}
Email: {email}
Check-in: {checkin_date}
Check-out: {checkout_date}
Number of guests: {num_guests}
House: {property_id}
Price per night: {price_per_night}
Total price: {total_price}

Step 2: If the user has confirmed to palce the booking, you tell him that the reservation is booked for 3 hours \ 
and that you will send an email with information about the booking along with payment instructions.

Step 3: If the user does not want place the booking, thank him and let him know that you are there if it needs anything else.

Here is a list of the last messages you exchanged with the user: 
{chat_history}

{format_instructions}"""

response_schemas = [
    ResponseSchema(name="booking_placed", description="True if the user decided to place the booking, False is the user does not want to place the booking, If the user has not answered yet return an empty string"),
    ResponseSchema(name="text", description="The response to the user"),
]


class BookingConfirmationChain:

    def __init__(self):

        llm = ChatOpenAI(temperature=0.)
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions =self.output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            input_variables=["user_name", "email", "checkin_date", "checkout_date", "num_guests", "property_id", "chat_history", "price_per_night", "total_price"], 
            partial_variables={"format_instructions": format_instructions},
            template=template
        )

        self.chain = LLMChain(llm=llm, 
                              prompt=prompt_template, 
                              verbose=chain_verbose,
                              output_key="booking_confirmation_info")

    def run(self, booking_info, chat_history):
        info = self.chain({"chat_history": chat_history,
                           "user_name": booking_info["user_name"], 
                           "email": booking_info["email"], 
                           "checkin_date": booking_info["check_in_date"], 
                           "checkout_date": booking_info["check_out_date"], 
                           "num_guests": booking_info["num_guests"], 
                           "property_id": booking_info["property_id"],
                           "price_per_night": booking_info["price_per_night"],
                           "total_price": booking_info["total_price"]})
        return self.output_parser.parse(info["booking_confirmation_info"])
    
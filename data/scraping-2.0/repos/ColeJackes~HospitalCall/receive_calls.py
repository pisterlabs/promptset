import logging
import typing
import os
from fastapi import FastAPI
from vocode.streaming.models.events import Event, EventType, PhoneCallConnectedEvent
from vocode.streaming.models.transcript import TranscriptCompleteEvent
from vocode.streaming.utils import events_manager
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TwilioInboundCallConfig, TelephonyServer
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.synthesizer import StreamElementsSynthesizerConfig
from dotenv import load_dotenv
from twilio.rest import Client
from string import ascii_uppercase
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load available times and match each one with a letter ( A), B), C) etc ) to make each option easier to distinguish
available_times = [time.rstrip() for time in open("available_times.txt", "r").readlines()]
letter_options = ascii_uppercase[:len(available_times)]
available_times_string = ''.join([a + ") " + b.rstrip() + ", " for a, b in zip(letter_options[:-1], available_times[:-1])])
available_times_string += 'or ' + letter_options[-1] + ") " + available_times[-1] + "?"

# Initialize Redis, Twilio, ChatGPT and Synthesizer, all to be used by TelephonyServer to handle incoming calls
config_manager = RedisConfigManager()
twilio_config = TwilioConfig(
    account_sid=os.environ["TWILIO_ACCOUNT_SID"],
    auth_token=os.environ["TWILIO_AUTH_TOKEN"],
)
twilio_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
agent_config = ChatGPTAgentConfig(
    prompt_preamble="Ask for each of the following pieces of information: 1. Full name, 2. Date of Birth, 3. Insurance "
                    "Payer Name, 4. Insurance ID, 5. If they have a referral, and which doctor they've been referred "
                    "to, 6. Reason for coming in, 7. Address, 8. Contact information. Ensure the user provides a valid "
                    "answer for each one, and ask again if necessary. Then ask which of these times they'd prefer: {} "
                    "Any of the listed time slots is valid and available. Once they've selected the time, thank them "
                    "and end the call.".format(available_times_string),
    initial_message=BaseMessage(
        text="Hello and welcome to Cole's Deluxe Hospital of Extreme Health. What is your name?"),
)
synth_config = StreamElementsSynthesizerConfig.from_telephone_output_device()

# Initialize Langchain chain to later extract user's selection of time slot for their appointment
llm = ChatOpenAI(temperature=0, model="gpt-4")
schema = {
    "properties": {
        # Currently this just extracts the user's time selection, since that is all that's needed for the follow-up text
        # I'm not sure this is the most robust way to extract the user's selection, but it never failed during testing
        # This approach could also be expanded to extract the other data (name, referral, reason, insurance etc)
        "time_choice_letter": {"type": "string"},
    },
    "required": ["time_choice_letter"],
}
chain = create_extraction_chain(schema, llm)

# Create dict to temporarily store link between conversation_id and its corresponding phone_number
phone_numbers_dict = dict()

# Initialize EventsManager to handle events when call connects and once the transcript is complete after the call ends
class EventsManager(events_manager.EventsManager):
    def __init__(self):
        super().__init__(subscriptions=[EventType.TRANSCRIPT_COMPLETE, EventType.PHONE_CALL_CONNECTED])

    async def handle_event(self, event: Event):
        if event.type == EventType.PHONE_CALL_CONNECTED:
            phone_call_connected_event = typing.cast(PhoneCallConnectedEvent, event)
            # Ideally this data linking from_phones to conversation_ids would just be taken directly from Redis,
            # where it is temporarily saved, but Vocode deletes it when each call ends.
            # See https://github.com/vocodedev/vocode-python/blob/727006fbb15c0dc897aaccbb2d3117bab52214eb/vocode/streaming/telephony/conversation/vonage_call.py#L121
            # So for now I am just storing the phone numbers with conversation_ids locally in a dict to guarantee nobody
            # is texted someone else's info
            phone_numbers_dict[event.conversation_id] = phone_call_connected_event.from_phone_number

        if event.type == EventType.TRANSCRIPT_COMPLETE:
            # Use Langchain to get user's selected time slot info from the transcript
            transcript_complete_event = typing.cast(TranscriptCompleteEvent, event)
            transcript = transcript_complete_event.transcript.to_string()
            selected_time = False
            successfully_booked = False
            try:
                letter = chain.run(transcript)[0]['time_choice_letter']
                index = ascii_uppercase.index(letter)
                selected_time = available_times[index].rstrip()
            except:
                twilio_client.messages \
                    .create(
                    body="Hello, we were unable to figure out which appointment slot you'd prefer. "
                         "Please call back and try again.",
                    from_=os.environ["OUTBOUND_CALLER_NUMBER"],
                    to=phone_numbers_dict[event.conversation_id]
                )

            if selected_time:
                # At this point, in the completed app, the user's appointment booking should be created
                successfully_booked = True

            if successfully_booked:
                twilio_client.messages \
                    .create(
                    body="We have successfully booked your appointment on {}.".format(selected_time),
                    from_=os.environ["OUTBOUND_CALLER_NUMBER"],
                    to=phone_numbers_dict[event.conversation_id]
                )

# Instantiate server to receive incoming calls
telephony_server = TelephonyServer(
    base_url=os.environ["TELEPHONY_SERVER_BASE_URL"],
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(url="/inbound_call",
                          agent_config=agent_config,
                          twilio_config=twilio_config,
                          synthesizer_config=synth_config)
    ],
    events_manager=EventsManager(),
    logger=logger,
)

# Run telephony_server using FastAPI
app = FastAPI(docs_url=None)
app.include_router(telephony_server.get_router())

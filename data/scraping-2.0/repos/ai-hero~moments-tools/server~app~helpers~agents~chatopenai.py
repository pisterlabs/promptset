import re
import sys
import logging
import pytz
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from moments.agent import Agent
from moments.moment import (
    Moment,
    Participant,
    Self,
    Context,
    Instructions,
    Example,
    Begin,
    Rejected,
)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s"
)
LOG = logging.getLogger(__name__)

chat = ChatOpenAI(temperature=0)


class ChatOpenAiAgent(Agent):
    def before(self: "ChatOpenAiAgent", moment: Moment):
        # Add context if not already present.
        is_context_added = False
        for occurrence in moment.occurrences:
            if isinstance(occurrence, Context):
                is_context_added = True
                break

        if not is_context_added:
            tz = pytz.timezone("America/Los_Angeles")
            current_time = datetime.now(tz)
            moment.occurrences.append(
                Context('```time: "' + current_time.strftime("%I:%M %p") + '"```')
            )

    def do(self: "ChatOpenAiAgent", moment: Moment):
        langchain_messages: list[BaseMessage] = []

        system = ""
        for occurrence in moment.occurrences:
            if (
                isinstance(occurrence, Instructions)
                or isinstance(occurrence, Example)
                or isinstance(occurrence, Begin)
            ):
                system += str(occurrence) + "\n"

        langchain_messages.append(SystemMessage(content=str(system)))
        for occurrence in moment.occurrences:
            if isinstance(occurrence, Self):
                langchain_messages.append(AIMessage(content=occurrence.content["says"]))
            elif isinstance(occurrence, Participant):
                langchain_messages.append(
                    HumanMessage(content=occurrence.content["says"])
                )

        # Complete with langchain
        response = chat(langchain_messages)
        line = response.content.splitlines()[0].strip()
        print(f"-->{line}<--")
        if not re.match(r"^Self:\s+(\((.*)\)\s+)?\"(.+)\"$", line):
            # Try to fix it
            if not line.startswith('"'):
                line = '"' + line
            if not line.endswith('"'):
                line = line + '"'
            if not line.startswith("Self: "):
                line = "Self: " + line

        # Get the rejected
        response = chat(langchain_messages)
        rejected_line = response.content.splitlines()[0].strip()
        rejected_line = rejected_line.replace("Self: ", "Rejected: ")
        if not re.match(r"^Self:\s+(\((.*)\)\s+)?\"(.+)\"$", rejected_line):
            # Try to fix it
            if not rejected_line.startswith('"'):
                rejected_line = '"' + rejected_line
            if not rejected_line.endswith('"'):
                rejected_line = rejected_line + '"'
            if not rejected_line.startswith("Rejected: "):
                rejected_line = "Rejected: " + rejected_line

        # First, add the rejected
        print(f"-->{rejected_line}<--")
        moment.occurrences.append(Rejected.parse(rejected_line))

        # Then the response
        moment.occurrences.append(Self.parse(line))

    def after(self: "ChatOpenAiAgent", moment: Moment):
        pass

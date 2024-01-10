from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI
from dotenv import dotenv_values


class Gptc:
    """
    Generative pre-trained traffic controller.
    This class takes in air traffic data, generates a natural-language representation, and outputs a traffic control command.
    """

    def __init__(self):
        # Load the model
        # Load environment file for secrets.
        secrets = dotenv_values(
            ".env"
        )  # Place .env file in the same directory as this file.
        # Define llm parameters
        self.llm = AzureChatOpenAI(
            deployment_name=secrets["model"],  # e.g. gpt-35-turbo
            openai_api_version=secrets["API_VERSION"],  # e.g. 2023-05-15
            openai_api_key=secrets["OPENAI_API_KEY"],  # secret
            azure_endpoint=secrets["azure_endpoint"],  # a URL
            # U-M shortcode
            openai_organization=secrets["OPENAI_organization"],
        )
        self.mode = "spd"  # Alt, hdg, or both.
        if self.mode == "alt":
            # Load prompt from file.
            with open("prompts/alt_prompt.txt", "r", encoding='utf-8') as f:
                self.prompt_header = f.read()
        elif self.mode == "hdg":
            with open("prompts/hdg_prompt.txt", "r", encoding='utf-8') as f:
                self.prompt_header = f.read()
        elif self.mode == "spd":
            with open("prompts/spd_prompt.txt", "r", encoding='utf-8') as f:
                self.prompt_header = f.read()
        else:
            self.prompt_header = "Act as an air traffic controller. \
                                    Your job is to issue a command to each aircraft, helping them avoid collisions. \
                                    Keep responses short and in the format <aircraft>: <heading> <flight level> <latitude> <longitude> \n"
        self.retry_message = "Please try again. Keep responses short and in the format <command> <aircraft> <value>. Give one line per aircraft."
        self.max_retry = 2

    def lon_to_ft(self, lon):
        """Convert longitude degrees to feet."""
        # This is an approximation that works for the US for differences between two longitudes.
        return lon * 268_560.0

    def lat_to_ft(self, lat):
        """Convert latitude degrees to feet."""
        # This is an approximation that works for differences between two latitudes (anywhere on the Earth's surface).
        return lat * 364_488.0

    def ms_to_knots(self, ms):
        """Convert m/s to knots."""
        # This is an approximation.
        return ms * 1.94384

    def m_to_ft(self, m):
        """Convert meters to feet."""
        # This is an approximation.
        return m * 3.28084

    def parse_radar_data(self, data):
        """
        Parse the air traffic data.
        Data is given as a dictionary with the following keys:
            - id: the aircraft id
        And the following values:
            - lat: latitude in degrees
            - lon: longitude in degrees
            - hdg: heading in degrees
            - alt: altitude in m
            - gs: ground speed in m/s 
            - vs: vertical speed in m/s
        Generate a natural-language representation of the air traffic data.
        """
        parsed_data = ""
        for id in data:
            parsed_data += f"Aircraft {id} is at lat {data[id]['lat']:.4f}, \
lon {data[id]['lon']:.4f} with heading {data[id]['hdg']:.1f} at altitude {self.m_to_ft(data[id]['alt']):.0f} ft. \
{id} has a groundspeed of {self.ms_to_knots(data[id]['gs']):.3f} knots and vertical speed of {self.m_to_ft(data[id]['vs'])*60:.3f} ft/min\n"
        if len(data) == 2:
            ac1 = list(data.keys())[0]
            ac2 = list(data.keys())[1]
            # Calculate the distance between the two aircraft.
            lat_dist = self.lat_to_ft(data[ac1]["lat"] - data[ac2]["lat"])
            lon_dist = self.lon_to_ft(data[ac1]["lon"] - data[ac2]["lon"])
            parsed_data += f"The aircraft are approximately {abs(lon_dist):.3f} ft apart in longitude.\n"
            parsed_data += f"The aircraft are approximately {abs(lat_dist):.3f} ft apart in latitude.\n"
        return parsed_data

    def get_commands(self, data):
        """
        Takes in sim data and returns a command.
        """
        # Convert raw sim data to natural language.
        nl_data = self.parse_radar_data(data)
        # Assemble the prompt.
        prompt = self.prompt_header + nl_data
        print(f"Sending message to model: {prompt}")
        msg = HumanMessage(content=prompt)
        # Ask the query.
        response = self.llm(messages=[msg])
        # Check response meets the required format for sim.
        # Retry with error message if response is not valid.
        print(f"Received response from model: {response.content}")
        retry_count = 0
        while retry_count < self.max_retry:
            if self.response_valid(response.content):
                break
            else:
                print("Invalid response. Retrying...")
                response = self.llm(messages=[msg])
                retry_count += 1
        return response.content.split("\n")

    def response_valid(self, response):
        """
        Parse the response from the model.
        """
        lines = response.split("\n")
        cmd = None
        if self.mode == "alt" or self.mode == "hdg" or self.mode == "spd":
            cmd = self.mode.upper()
        if cmd is not None:
            for line in lines:
                if not line.startswith(cmd):
                    print(f"Line does not start with {cmd}.")
                    return False

        # Check that all lines are short enough.
        if self.mode == "alt" or self.mode == "hdg" or self.mode == "spd":
            max_line_length = 20
            for line in lines:
                if len(line) > max_line_length:
                    print("Line too long.")
                    return False
            return True
        else:
            line_count_valid = len(lines) == 3
            if not line_count_valid:
                print("Wrong number of lines.")
            line_length_valid = True
            for line in lines:
                if len(line) > 30:
                    print("Line too long.")
                    line_length_valid = False
            return line_count_valid and line_length_valid

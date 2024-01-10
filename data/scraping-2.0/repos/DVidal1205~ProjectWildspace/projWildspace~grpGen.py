from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate


class grp:

    def __init__(self, ui, chat, world):
        # Create UI
        self.ui = ui

        # Create World
        self.world = world

        # Create Chat Model
        self.chat_llm = chat

        # Generate Schema
        self.name = ResponseSchema(name="name", description="Faction Name. Be creative, and make the name sound fantasy by using prefixes from various languages")
        self.goal = ResponseSchema(name="goal", description="Common goal / cause of the faction (1-3 Sentences)")
        self.features = ResponseSchema(name="features", description="Identifying features of the faction, such as symbols, clothing, phrases, etc (1-3 Sentences)")
        self.morality = ResponseSchema(name="morality", description="Morality of the faction. Use D&D Alignments as responses")
        self.activities = ResponseSchema(name="activities", description="Activities / Presence of the faction (1-3 Sentences)")
        self.devotion = ResponseSchema(name="devotion", description="Level of devotion to the faction (ex. Low, Medium, High, Fanatic, etc)")
        self.population = ResponseSchema(name="population", description="The population of this faction (1 to 25,000 inclusive)")
        self.response_schema = [self.name, self.goal, self.features, self.morality, self.activities, self.devotion, self.population]

        # Create Schema Parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Create Template
        self.template_string = """You are an expert Dungeon Master for Dungeons and Dragons Fifth Edition \ 
        You come up with memorable and impactful groups and factions that your party may interact with.
        
        Create a group or faction that your party may stumble upon during their travels.

        When creating this group or faction, be sure to include the following world information in your worldbuilding process.

        {world_info}   
        
        {format_instructions}"""

    def generate(self):

        # Create Prompt
        self.prompt = ChatPromptTemplate.from_template(template=self.template_string)

        # Create Messages
        self.messages = self.prompt.format_messages(format_instructions=self.format_instructions, world_info=self.world.loadWorld(self))
        # Parse Response
        self.response = self.chat_llm(self.messages)
        self.response_as_dict = self.output_parser.parse(self.response.content)

        # Labels
        self.ui.grpCauseLabel.setText(self.response_as_dict["goal"])
        self.ui.grpNameLabel.setText(self.response_as_dict["name"])
        self.ui.grpPopTxt_2.setText(self.response_as_dict["population"])
        self.ui.grpDevotionTxt_2.setText(self.response_as_dict["devotion"])
        self.ui.grpMoralityLabel.setText(self.response_as_dict["morality"])
        self.ui.grpTraitsLabel.setText(self.response_as_dict["features"])
        self.ui.grpPresenceLabel.setText(self.response_as_dict["activities"])


    def save(self):

        # Create File
        self.filename = "saves/groups/" + self.response_as_dict["name"].lower().replace(" ", "_") + ".txt"
        self.file = open(self.filename, "w")

        # Write to File
        self.file.write("Name: " + self.response_as_dict["name"] + "\n")
        self.file.write("Number of Members: " + self.response_as_dict["population"] + "\n")
        self.file.write("Devotion: " + self.response_as_dict["devotion"] + "\n")
        self.file.write("Morality: " + self.response_as_dict["morality"] + "\n")
        self.file.write("Goal: " + self.response_as_dict["goal"] + "\n")
        self.file.write("Features: " + self.response_as_dict["features"] + "\n")
        self.file.write("Activities: " + self.response_as_dict["activities"] + "\n")

        # Close File
        self.file.close()


        

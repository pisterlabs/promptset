from common import *
import json
from pymongo import MongoClient
client = MongoClient()

storyline = list(client.lost_mines.story.find())
# NPCs = list(client.lost_mines.NPCs.find())

adventure_name = "alec_first"

#model = "gpt-3.5-turbo"
model = "gpt-4-1106-preview"



from modularity import OpenAI
import traceback

client = OpenAI()

class Convo:

    def __init__(self, story_name):

        self.story_name = story_name

        self.story_dir = Path(f"stories/{story_name}")
        self.story_dir.mkdir(exist_ok=True, parents=True)

        self.story_part_name = self.get_txt("story_part")
        if self.story_part_name is None:
            self.story_part_name = 'start'
            self.set_txt("story_part", self.story_part_name)

        dialogue = self.get_txt("dialogue")
        if dialogue is None:
            self.M = []
        else:
            self.M = [{"role": "user", "content": 'PREVIOUS DIALOGUE:\n' + dialogue[-1500:] + "..."}]
            self.briefly_summarize()

        self.summary = []
        self.story_part_name = 'start'
        self.type_modifiers = {
            'strength': 2,
            'dexterity': 1,
            'constitution': 0,
            'intelligence': -1,
            'wisdom': -2,
            'charisma': -3,
        }

    @property
    def story_part(self):
        return [x for x in storyline if x['name'] == self.story_part_name][0]

    def briefly_summarize(self):
        self.computersay(f"(summarizing from last time...) " + self.summarize_plotline("Explain this to the player, bringing them up to speed on what just happened. Hopefully just one sentence will suffice."))

    def get_txt(self, name):
        story_part_file = self.story_dir / f"{name}.txt"
        if story_part_file.exists():
            return story_part_file.read_text().strip()
        else:
            return None

    def set_txt(self, name, content):
        f = self.story_dir / f"{name}.txt"
        f.write_text(content)

    def add_txt(self, name, content):
        f = self.story_dir / f"{name}.txt"
        if not f.exists():
            f.write_text(content)
        else:
            f.write_text(f.read_text() + "\n" + content)

    # ------------------
    # Summarizing is probably useful!
    # ------------------

    def summarize_character(self):
        message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in self.M])
        prompt = f"""
        Your goal is to extract full character sheets for the players involved in this adventure.

        Messages:
        {message_text}
        """
        print(prompt)

        messages = [
            {"role": "system", "content": prompt},
        ]

        response = get_response(messages, model=model)
        return response

    def summarize_plotline(self, prompt=None):
        message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in self.M])
        if prompt is None:
            prompt = f"""
            Your goal is to summarize the plotpoints contained in the following conversation between a DM and a player.
            In each plot point, be as specific as possible.
            Keep note of any characters, locations, or items that are mentioned.
            Do not include any information not present in the following messages!
            Please be extremely concise.
            """

        prompt += f"""

            Messages:
            {message_text}
        """
        #print(prompt)

        messages = [
            {"role": "system", "content": prompt},
        ]

        response = get_response(messages, model=model)
        #print('Summarized!')
        #print(response)
        return response

    def inventory(self, action, object):
        self.add_txt("inventory", f"{action}: {object}")
        return f"Inventory {action}: {object}"
    
    def change_scene(self, to):
        self.story_part_name = to
        self.set_txt("story_part", self.story_part_name)

        return "Changed scene to " + to
    
    def roll_hit_dice(self, n_sides, n_dice, type=None, **kwargs):
        import random
        result = [ random.randint(1, n_sides) for i in range(n_dice) ]
        result = result_og = sum(result)
        mod = 0
        if type is not None and type in self.type_modifiers:
            mod += self.type_modifiers[type]
        result += mod

        return f"Rolled {n_dice}d{n_sides} ({type}) {result_og} + {mod} = {result}"
    

    # ------------------
    # SAYING STUFF
    # ------------------

    def humansay(self, content):
        self.M.append({"role": "user", "content": content})
        self.add_txt("dialogue", f"Player:\n{content}")

    def computersay(self, content):
        self.M.append({"role": "assistant", "content": content})
        self.add_txt("dialogue", f"DM:\n{content}")
        print("DM:", content)

    def computersay_self(self, content):
        self.M.append({"role": "system", "content": content})
        self.add_txt("dialogue", f"DM (to themselves):\n{content}")

    def _format_inventory(self):

        inventory = self.get_txt("inventory")
        if inventory is None:
            if self.story_part_name == 'start':
                self.inventory("add", "10 gold pieces")
                self.inventory("add", "a backpack")
                self.inventory("add", "a bedroll")
                self.inventory("add", "a mess kit")
                self.inventory("add", "a tinderbox")
                self.inventory("add", "10 torches")
                self.inventory("add", "10 days of rations")
                self.inventory("add", "a waterskin")
                self.inventory("add", "50 feet of hempen rope")
                return self._format_inventory()
            else:
                inventory = "The player has nothing."

        return inventory
    
    # ------------------
    # Thinking, Acting, and Responding
    # ------------------

    def think(self):
        prompt = f"""
        You are an assistant to the DM.
        Speak directly to the DM (not the player).
        Give some thoughts or ideas to the DM to help them conduct their duties.
        If you think everything is clear, type 'pass'.
        Be concise, specific, and clear.
        """

        messages = [
            {"role": "system", "content": prompt},
            *self.M,
            {"role": "system", "content": "What do you think to yourself? Be brief."},
        ]

        response = get_response(messages, model=model)

        self.computersay_self("(thinking...) " + response)

    def act(self):
        story_part = self.story_part

        next_steps = "\n".join([f"\t{x}: {y}" for x, y in story_part['next_steps'].items()])
        inventory = self._format_inventory()
        prompt = f"""
            Your current inventory is:
            {inventory}
        
            Based on the dialogue so far, you are to decide what actions to take next.
            Most of the time no action will need to be taken. In this case, simply type "pass".
            Please do not act in a way not directly implied by the dialogue so far.
            Although there is no rush to change the 'scene', you must eventually do so, in order to continue the story.

            If you want to change the scene, type:
            {{"type": "change_scene", "to": "scene name"}}

            To roll hit dice, type:
            {{"type: "roll_hit_dice", "n_dice": 1, "n_sides": 6, "type": "strength"}}

            To add or remove from inventory, type:
            {{"type: "inventory", "action": "add|remove", "object": "object name, description, and/or stats"}}

            ALWAYS USE DOUBLE QUOTES FOR JSON STRINGS

            You can type a command on each line.
            You CANNOT mix commands and statements.

            Scenes available, their names and descriptions:
            {next_steps}
        """

        messages = [
            {"role": "system", "content": prompt},
            *self.M,
            {"role": "system", "content": "What do you do? (type = change_scene, roll_hit_dice, or inventory). Use only JSON strings, one per line. If no action need be taken from the most recent message, simply type 'pass'."},
        ]

        response = get_response(messages, model=model)
        if response.strip() == "pass":
            return

        parts = response.split("}")
        for part in parts:
            if part == "":
                continue
            part += "}"

            try:
                part = json.loads(part)
                self.act_on(part)
            except json.decoder.JSONDecodeError:
                print("Invalid JSON:", part)

    def act_on(self, action):
        print("Executing... ", json.dumps(action, indent=2))
        act = dict(action)
        typ = action.pop('type')

        try:
            fn = getattr(self, typ)
            response = fn(**action)
            self.computersay_self(response)
        except Exception as e:
            # first get the last line of the traceback
            tb = traceback.format_exc().splitlines()[-1]

            # then get the last line of the error
            error = str(e).splitlines()[-1]

            self.computersay_self(f"Error in command '{json.dumps(act, indent=2)}': {error} ({tb})")
            self.computersay_self("Please rewrite this one.")
            self.act()

    def respond(self):

        story_part = self.story_part

        my_messages = []
        inventory = self._format_inventory()
        prompt = f"""
        You are a DM. 
        You are currently in the scene "{story_part['name']}".

        Your current inventory is:
        {inventory}

        The message you type will be sent to the player from the DM.

        Description of the current scene:
            {story_part['description']}
        """
        
        my_messages.append({'role': 'system', 'content': prompt})

        response = get_response(my_messages + self.M, model=model)

        self.computersay(response)

        # consolidate things, if it's getting long
        if len(self.M) > 10:
            # remember a summary of the messages
            self.summary.append(self.summarize_plotline())

            # (mostly) clear the messages
            self.M = self.M[-2:]


    # ------------------
    # Running the Conversation
    # ------------------


    def run(self):
        # human does its thing
        query = input(">> ")
        self.humansay(query)

        # computer does its thing
        self.act()
        self.think()
        self.respond()
        self.act()

    def loop(self):
        while True:
            self.run()


c = Convo(adventure_name)
c.loop()
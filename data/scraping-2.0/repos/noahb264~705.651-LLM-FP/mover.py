from abc import abstractmethod
import random
from render import display_level_with_labels, label_map
from io import BytesIO
from base64 import b64encode
import openai
import matplotlib.pyplot as plt
import json
import time

class Mover():

    @abstractmethod
    def get_next_move(self, current_lvl):
        pass

class RandomMover(Mover):

    def get_next_move(self, current_lvl, current_plr, all_moves, all_comms):
        move = random.choice(["move right", "move left", "jump on right", "jump on left"])
        return {current_plr: move}, {current_plr: ""}
    
class FileMover(Mover):

    def __init__(self):
        with open("./moves.txt", "r") as f:
            self.moves = json.loads(f.read())

    def get_next_move(self, current_lvl, current_plr, all_moves, all_comms):
        return self.moves.pop(0), {current_plr: ""}

class TextMover(Mover):
    
    def __init__(self):
        api_key = open("./api_key.txt", "r").read().strip()
        self.client = openai.OpenAI(api_key=api_key)

    def get_next_move(self, current_lvl, current_plr, all_moves, all_comms):
        all_moves = "\n".join([f"- {p}: {m}" for p, m in all_moves])
        all_comms = "\n".join([f"- {p}: {c}" for p, c in all_comms])

        with open('pico_prompt.txt', 'r') as f:
            task_prompt = f.read()

        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()

        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            # response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt.format(player=current_plr)},
                {"role": "user", "content": task_prompt.format(level=current_lvl,
                                                            moves=all_moves,
                                                            thoughts=all_comms,
                                                            player=current_plr)}
            ]
        )

        moves = response.choices[0].message.content
        print("\n", moves, "\n")
        if "{" not in moves or "}" not in moves:
            raise Exception(f"Invalid response from LLM: {moves}")

        move_idx_start = moves.index("{")
        move_idx_end = moves.index("}", move_idx_start + 1)
        move = json.loads(moves[move_idx_start:move_idx_end+1])
        # print(move["move"])
        time.sleep(random.random()*10)
        return {current_plr: move["move"]}, {current_plr: move["communication"]}

class VisionMover(Mover):
    
    def __init__(self):
        api_key = open("./api_key.txt", "r").read().strip()
        self.client = openai.OpenAI(api_key=api_key)

    def get_next_move(self, current_lvl, current_plr, all_moves, all_comms):
        all_moves = "\n".join([f"- {label_map[p]}: {m}" for p, m in all_moves])
        all_comms = "\n".join([f"- {label_map[p]}: {c}" for p, c in all_comms])

        with open('vision_pico_prompt.txt', 'r') as f:
            task_prompt = f.read()

        with open('vision_system_prompt.txt', 'r') as f:
            system_prompt = f.read()

        fig = display_level_with_labels(current_lvl)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        image = b64encode(buf.read()).decode('ascii')

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt.format(player=label_map[current_plr])},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": task_prompt.format(level=current_lvl, moves=all_moves, thoughts=all_comms, player=label_map[current_plr])
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        moves = response.choices[0].message.content
        print("\n", moves, "\n")
        if "{" not in moves or "}" not in moves:
            raise Exception(f"Invalid response from LLM: {moves}")
        move_idx_start = moves.index("{")
        move_idx_end = moves.index("}", move_idx_start + 1)
        move = json.loads(moves[move_idx_start:move_idx_end+1])
        print(move["move"])
        time.sleep(random.random()*20)
        return {current_plr: move["move"]}, {current_plr: move["communication"]}
        

MOVERS = {
    "random": RandomMover,
    "file": FileMover,
    "text": TextMover,
    "vision": VisionMover,
}
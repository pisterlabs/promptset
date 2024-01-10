import json
from pathlib import Path

import openai


PROMPT = "A Devil over the forest and brave elf warrior protecting the viever, first person hands, stack of melee weapons, grimdark fantasy realistic painting style"
# PROMPT = "An old Wizard in big round hat standing next to a vending machine in a cyberpunk japanese world, spaceport outside the window, grimdark fantasy realistic painting style"
# PROMPT = "Deus Ex Mankind Divided syle victorian industrial hall bacground, cybernetic black shiny prosthetic hand, double helix of red filament out of printer, shoggoth"
# PROMPT = "H. R. Giger style relief, a desperate mother at the gates of Tartar with his faithful wolf companion"

DATA_DIR = Path.cwd() / "responses"
DATA_DIR.mkdir(exist_ok=True)


with open('key.txt', 'r') as f:
    openai.api_key = f.read()


response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size="1024x1024",
    response_format="b64_json",
)


file_name = DATA_DIR / f"{PROMPT[:8]}-{response['created']}.json"

with open(file_name, mode="w", encoding="utf-8") as file:
    json.dump(response, file)

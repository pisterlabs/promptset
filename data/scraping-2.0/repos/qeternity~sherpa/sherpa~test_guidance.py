import sys

sys.path.append("/root/sherpa/exllamav2")
sys.path.append("/root/sherpa/guidance")
import guidance
from transformers import AutoConfig, AutoTokenizer

model_path = "/root/Nous-Hermes-Llama2-13b-GPTQ"
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from exllama_hf import ExllamaHF

model = ExllamaHF.from_pretrained(model_path)

# from exllamav2_hf import Exllamav2HF
# model = Exllamav2HF.from_pretrained(model_path)

model.config = config

guidance.llm = guidance.llms.Transformers(
    model,
    tokenizer,
    caching=False,
    acceleration=False,
    device=0,
)

generated = guidance(
    """The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=128}}"""
)()
print(generated)

# we can pre-define valid option sets
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]

# define the prompt
character_maker = guidance(
    """The following is a character profile for an RPG game in JSON format.
```json
{
    "id": "{{id}}",
    "description": "{{description}}",
    "name": "{{gen 'name'}}",
    "age": {{gen 'age' pattern='[0-9]+' stop=','}},
    "armor": "{{#select 'armor'}}leather{{or}}chainmail{{or}}plate{{/select}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}",
    "class": "{{gen 'class'}}",
    "mantra": "{{gen 'mantra' temperature=0.7}}",
    "strength": {{gen 'strength' pattern='[0-9]+' stop=','}},
    "items": [{{#geneach 'items' num_iterations=5 join=', '}}"{{gen 'this' temperature=0.7}}"{{/geneach}}]
}```"""
)

# generate a character
generated = character_maker(
    id="e1f491f7-7ab8-4dac-8c20-c92b5e7d883d",
    description="A quick and nimble fighter.",
    valid_weapons=valid_weapons,
)
print(generated)

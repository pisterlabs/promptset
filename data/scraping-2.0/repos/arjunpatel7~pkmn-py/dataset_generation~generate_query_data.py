import random
from langchain.llms import Cohere
import jsonlines

# This file will be exactly the same as generate_speed_data.py, except for the
# the prompts being related to the query task instead of the speed task.
from generate_speed_data import random_mon


pokemons = []
with jsonlines.open("./data/gen9_pokemon.jsonl") as reader:
    for entry in reader:
        pokemons.append(entry)


def random_stat_name():
    return random.choice(
        ["attack", "defense", "special attack", "special defense", "speed"]
    )


# first section of prompts
# these simply ask the stat of a pokemon
stat_templates = [
    "What is the {stat} stat of {pokemon}?",
    "Tell me the {stat} of {pokemon}?",
    "{pokemon}'s {stat} stat is what value?",
    "What is {pokemon}'s {stat} stat?",
]

# second section of prompts
# these ask about the top X pokemon in a stat
top_x_templates = [
    "What are the top {x} pokemon with the highest {stat}?",
    "Which pokemon are the top {x} for the {stat} stat?",
]


# third section of prompts
# these ask about the bottom X pokemon in a stat

bottom_x_templates = [
    "What are the bottom {x} pokemon with the lowest {stat}?",
    "Who are the bottom {x} pokemon for the {stat} stat?",
]


# fourth section of prompts
# these are random prompts that are unrelated to querying a pokemon's stats

# we'll use a Cohere generate endpoint to create the prompts and write them out


llm = Cohere(model="command")


def create_random_pokemon_prompts(x):
    prompts = []
    for i in range(0, x):
        new_prompt = llm(
            "Come up with a one sentence question related to the pokemon video game."
        )
        prompts.append(new_prompt)
    return prompts


def create_dataset(num_samples=500):
    # randomly samples the prompt templates
    # fills in with random pokemon names and stat names and numbers
    # returns a list of prompts

    # randomly creates pokemon prompts, equal number

    # writes all out in jsonl file
    all_prompts = []

    for x in range(0, 250):
        # do the first section of prompts
        # randomly sample a template
        template = random.choice(stat_templates)
        # randomly sample a pokemon
        pokemon = random_mon(pokemons)
        # randomly sample a stat
        stat = random_stat_name()
        # fill in the template
        prompt = template.format(pokemon=pokemon, stat=stat)
        all_prompts.append(prompt)
    for x in range(0, 250):
        # do the second section of prompts
        # randomly sample a template
        template = random.choice(top_x_templates)
        # randomly sample a pokemon
        pokemon = random_mon(pokemons)
        # randomly sample a stat
        stat = random_stat_name()
        # randomly sample a number
        x = random.randint(1, 10)
        # fill in the template
        prompt = template.format(pokemon=pokemon, stat=stat, x=x)
        all_prompts.append(prompt)
    for x in range(0, 250):
        # do the third section of prompts
        # randomly sample a template
        template = random.choice(bottom_x_templates)
        # randomly sample a pokemon
        pokemon = random_mon(pokemons)
        # randomly sample a stat
        stat = random_stat_name()
        # randomly sample a number
        x = random.randint(1, 10)
        # fill in the template
        prompt = template.format(pokemon=pokemon, stat=stat, x=x)
        all_prompts.append(prompt)

    # get random prompts from the Cohere API
    random_prompts = create_random_pokemon_prompts(50)
    # combine all prompts
    all_prompts.extend(random_prompts)
    # write out to jsonl file
    with jsonlines.open("./data/query_prompts.jsonl", "w") as writer:
        print("Writing prompts....")
        for prompt in all_prompts:
            writer.write(prompt)


if __name__ == "__main__":
    create_dataset()

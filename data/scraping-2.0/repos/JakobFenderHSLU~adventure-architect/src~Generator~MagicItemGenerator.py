import json
import random

from Utility import OpenAIConnection
from Utility.ItemHelper import item_template, possible_armor_subtypes, possible_item_rarities, possible_item_types, \
    possible_weapons_subtypes


def generate_magic_item(description, selected_rarity, selected_type, requires_attunement, cursed):
    chatgpt_messages = generate_magic_item_prompt(description, selected_rarity, selected_type, requires_attunement,
                                                  cursed)
    return OpenAIConnection.generate_text(chatgpt_messages)


def generate_magic_item_prompt(description, selected_rarity, selected_type, requires_attunement, cursed):
    return_format = f"""
    THE RETURN FORMAT SHOULD ALWAYS BE PARSABLE TO JSON. 
    DO NOT USE LINE BREAKS. START WITH "{{" AND END WITH "}}". \n
    Return the Item in the following format: \n
    {item_template}
    """

    attribute_explanation = f"""
    "Subtype": Armor and Weapons have subtypes. Choose between the following for Armor:
     {", ".join(possible_armor_subtypes)} \n 
     And choose between the following for Weapons: {", ".join(possible_weapons_subtypes)} \n
    "Attunement": Choose between Yes and No. Scrolls and Potions can not be attuned. \n
    "Visual Description": Write a description of the item. Make it as detailed as possible.\n
    "Mechanical Description": Write a description of the item. This is the description that will be used in combat.
    This attribute needs to be filled out! \n
    "Story": Write a story about the item. This attribute is optional. \n
    "Price": Write the price of the item in gp. \n
    "Cursed": A description of the curse. This attribute needs to be filled out, if the item is cursed. \n
    """

    rarity_for_request = selected_rarity
    type_for_request = selected_type
    attunement_for_request = requires_attunement
    cursed_for_request = cursed

    if selected_rarity == "Random":
        rarity_for_request = random.choice(possible_item_rarities)
    if selected_type == "Random":
        type_for_request = random.choice(possible_item_types)
    if requires_attunement == "Random":
        attunement_for_request = random.choice(["Yes", "No"])
    if cursed == "Random":
        cursed_for_request = random.choice(["Yes", "No"])

    guide = """
    Here is a quick guide on how to create a Magic Item: \n
    
    Magic Item Rarity and recommended price: \n
    Rarity      Value \n
    Common	    50-100 gp \n
    Uncommon	101-500 gp \n
    Rare        501-5,000 gp \n
    Very rare   5,001-50,000 gp \n
    Legendary	50,001+ gp \n
    
    Magic Bonus and Rarity: \n
    Rarity      Bonus \n
    Rare        +1 \n
    Very rare   +2 \n
    Legendary	+3 \n
    Strictly stick to the recommended Magic Bonus. \n
    
    Spell Level and Rarity: \n
    Rarity	    Max Spell Level
    Common	    1st
    Uncommon	3rd
    Rare	    6th
    Very rare	8th
    Legendary	9th
    
    """

    prompt = f"""
    Create an Magic Item for Dungeons and Dragons. According to the previous format. Here is what I had in mind: \n
    Rarity: {rarity_for_request} \n
    ItemType: {type_for_request} \n
    Requires Attunement: {attunement_for_request} \n
    Item is cursed: {cursed_for_request} \n

    Description always has priority over the other attributes. \n
    Description: {description}
    """

    chatgpt_messages = [
        {"role": "system", "content": "Create an Magic Item for Dungeons and Dragons."},
        {"role": "system", "content": return_format},
        {"role": "system", "content": attribute_explanation},
        {"role": "system", "content": guide},
        {"role": "user", "content": prompt}
    ]

    print(chatgpt_messages)

    return chatgpt_messages

import json
import random
import pandas as pd

from Utility import OpenAIConnection
from Utility.MonsterHelper import possible_monster_types, possible_challenge_ratings, possible_size, \
    possible_environments, monster_template

balance_data = pd.read_csv("assets/balance_stats.csv")


def generate_monster(description, selected_monster_type, selected_challenge_rating, selected_size, selected_environment,
                     selected_legendary, selected_lair):
    chatgpt_messages = generate_monster_prompt(description, selected_monster_type, selected_challenge_rating,
                                               selected_size, selected_environment, selected_legendary, selected_lair)
    return OpenAIConnection.generate_text(chatgpt_messages)


def generate_monster_prompt(description, selected_monster_type, selected_challenge_rating, selected_size,
                            selected_environment, selected_legendary, selected_lair):
    return_format = f"""
    THE RETURN FORMAT SHOULD ALWAYS BE PARSABLE TO JSON. 
    DO NOT USE LINE BREAKS. START WITH "{{" AND END WITH "}}". \n
    Return the Monster in the following format: : \n
    {monster_template}
    """

    monster_type_for_request = selected_monster_type
    challenge_rating_for_request = selected_challenge_rating
    size_for_request = selected_size
    environment_for_request = selected_environment
    legendary_for_request = selected_legendary
    lair_for_request = selected_lair

    if selected_monster_type == "Random":
        monster_type_for_request = random.choice(possible_monster_types)
    if selected_challenge_rating == "Random":
        challenge_rating_for_request = random.choice(possible_challenge_ratings)
    if selected_size == "Random":
        size_for_request = random.choice(possible_size)
    if selected_environment == "Random":
        environment_for_request = random.choice(possible_environments)
    if selected_legendary == "Random":
        legendary_for_request = random.choice(["Yes", "No"])
    if selected_lair == "Random":
        lair_for_request = random.choice(["Yes", "No"])

    average_stats = balance_data[balance_data['CR'] == challenge_rating_for_request].iloc[0]

    attribute_explanation = f"""
    "Type of Monster": Choose between {", ".join(possible_monster_types)} \n
    "Armor Class": The average Armor Class for your Challenge Rating is {average_stats["Armor Class"]}. 
    You can choose a different score!\n
    "Hit Points": The average Hit Points for your Challenge Rating is {average_stats["Hit Points"]}. 
    You can choose a different amount, if it makes sense!\n
    "Speed": Speed is usually 30 ft. Monsters can have mulitple speeds for different modes of transportation.
    Choose at least one between "Walk", "Fly", "Swim", "Climb", "Burrow". \n
    "Legendary Action Description": Only fill out, if "Legendary Actions" is not empty \n
    "Lair Action Description": Only fill out, if "Lair Actions" is not empty \n
    "Bonus Actions": A weaker version of "Actions". \n
    "Actions": Add actions that your NPC can take. The average Variables for your Challenge Rating is as follows: \n
    Hit Bonus: {average_stats["Attack Bonus"]} \n
    Damage per Round: {average_stats["Average DPR"]} \n
    Save DC: {average_stats["Save DC"]} \n
    You can choose different values, if it makes sense!
    """

    guide = """
    Here is a quick guide on how to create a Monster: \n

    Recommended Statistics per Challenge Rating: \n
    CR	Prof. Bonus	Armor Class 	Hit Points	Attack Bonus	Damage/Round	Save DC
    0	+2  	    ≤ 13	        1-6	≤       +3	            0-1	            ≤ 13
    1/8	+2	        13	            7-35	    +3	            2-3	            13
    1/4	+2	        13	            36-49	    +3	            4-5	            13
    1/2	+2	        13	            50-70	    +3	            6-8	            13
    1	+2	        13	            71-85	    +3	            9-14	        13
    2	+2	        13	            86-100	    +3	            15-20	        13
    3	+2	        13	            101-115	    +4	            21-26	        13
    4	+2	        14	            116-130	    +5	            27-32	        14
    5	+3	        15	            131-145	    +6	            33-38	        15
    6	+3	        15	            146-160	    +6	            39-44	        15
    7	+3	        15	            161-175	    +6	            45-50	        15
    8	+3	        16	            176-190	    +7	            51-56	        16
    9	+4	        16	            191-205	    +7	            57-62	        16
    10	+4	        17	            206-220	    +7	            63-68	        16
    11	+4	        17	            221-235	    +8	            69-74	        17
    12	+4	        17	            236-250	    +8	            75-80	        17
    13	+5	        18	            251-265	    +8	            81-86	        18
    14	+5	        18	            266-280	    +8	            87-92	        18
    15	+5	        18	            281-295	    +8	            93-98	        18
    16	+5	        18	            296-310	    +9	            99-104	        18
    17	+6	        19	            311-325	    +10	            105-110	        19
    18	+6	        19	            326-340	    +10	            111-116	        19
    19	+6	        19	            341-355	    +10	            117-122	        19
    20	+6	        19	            356-400	    +10	            123-140	        19
    21	+7	        19	            401-445	    +11	            141-158	        20
    22	+7	        19	            446-490	    +11	            159-176	        20
    23	+7	        19	            491-535	    +11	            177-194	        20
    24	+7	        19	            536-580	    +12	            195-212	        21
    25	+8	        19	            581-625	    +12	            213-230	        21
    26	+8	        19	            626-670	    +12	            231-248	        21
    27	+8	        19	            671-715	    +13	            249-266	        22
    28	+8	        19	            716-760	    +13	            267-284	        22
    29	+9	        19	            761-805	    +13	            285-302	        22
    30	+9	        19	            806-850	    +14	            303-320	        23
    """

    prompt = f"""
    Create an Monster for Dungeons and Dragons. According to the previous format. Here is what I had in mind: \n
    Type of Monster: {monster_type_for_request} \n
    Challenge Rating: {challenge_rating_for_request} \n
    Size: {size_for_request} \n
    Environment: {environment_for_request} \n
    Legendary: {legendary_for_request} \n
    Lair: {lair_for_request} \n

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

    return chatgpt_messages

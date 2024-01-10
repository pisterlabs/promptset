import json
import random

from Utility import OpenAIConnection
from Utility.SpellHelper import possible_conditions, possible_damage_types, possible_save_types, \
    possible_spell_area_types, possible_spell_casting_time, possible_spell_durations, possible_spell_levels, \
    possible_spell_ranges, possible_spell_schools, possible_spell_types, spell_template


def generate_spell(description, selected_level, selected_school, selected_range, selected_components,
                   selected_spell_type, selected_save_type, selected_spell_area, selected_requires_concentration,
                   selected_damage_type, selected_condition, selected_ritual, selected_casting_time,
                   selected_spell_duration):
    chatgpt_messages = generate_spell_prompt(description, selected_level, selected_school, selected_range,
                                             selected_components, selected_spell_type, selected_save_type,
                                             selected_spell_area, selected_requires_concentration,
                                             selected_damage_type, selected_condition, selected_ritual,
                                             selected_casting_time, selected_spell_duration)
    return OpenAIConnection.generate_text(chatgpt_messages)


def generate_spell_prompt(description, selected_level, selected_school, selected_range, selected_components,
                          selected_spell_type, selected_save_type, selected_spell_area, selected_requires_concentration,
                          selected_damage_type, selected_condition, selected_ritual, selected_casting_time,
                          selected_spell_duration):
    return_format = f"""
    THE RETURN FORMAT SHOULD ALWAYS BE PARSABLE TO JSON. 
    DO NOT USE LINE BREAKS. START WITH "{{" AND END WITH "}}". \n
    Return the Spell in the following format: : \n
    {spell_template}
    """

    attribute_explanation = f"""
        "Level": Choose between the following: {", ".join(possible_spell_levels)}. 
        This attribute needs to be filled out! \n
        "School": Choose between the following: {", ".join(possible_spell_schools)} 
        This attribute needs to be filled out! \n
        "Range": Choose between the following: {", ".join(possible_spell_ranges)} 
        This attribute needs to be filled out! \n
        "Spell Components": Choose one or multiple: "Verbal", "Somatic", "Material". 
        This attribute needs to be filled out! \n
        "Material Components": Components that are required to cast the spell. Choose random ingredients that are related 
        to the damage type, condition and/or description. This attribute is optional. \n
        "Casting Time": Choose between the following: {", ".join(possible_spell_casting_time)} 
        This attribute needs to be filled out! \n
        "Duration": Choose between the following: {", ".join(possible_spell_durations)} 
        This attribute needs to be filled out! \n
        "Requires Concentration": Choose between Yes and No. This attribute needs to be filled out! \n
        "Ritual": Choose between Yes and No. This attribute needs to be filled out! \n
        "Spell Type": Choose between the following: {", ".join(possible_spell_types)} 
        This attribute needs to be filled out! \n
        "Save Type": Choose between the following: {", ".join(possible_save_types)} This attribute is optional. \n
        "Spell Area": Choose between the following: {", ".join(possible_spell_area_types)} 
        This attribute is optional. \n
        "Damage Type": Choose between the following: {", ".join(possible_damage_types)} This attribute is optional. \n
        "Condition": Choose between the following: {", ".join(possible_conditions)} This attribute is optional. \n
        "Description": Write a description of the spell. Make it as detailed as possible. This attribute needs to be
        filled out! \n
        """

    guide = """
        Here is a quick guide on how to create a Spell: \n
        
        When creating a new spell, use existing spells as guidelines. Here are some things to consider:
        - If a spell is so good that a caster would want to use it all the time, it might be too powerful for its level.
        - A long duration or large area can make up for a lesser effect, depending on the spell.
        - Avoid spells that have very limited use, such as one that works only against good dragons. Though such a 
        spell could exist in the world, few characters will bother to learn or prepare it unless they know in advance 
        that doing so will be worthwhile.

        Spell Damage: \n
        For any spell that deals damage, use the Spell Damage table to determine approximately how much damage is 
        appropriate given the spell's level. The table assumes the spell deals half damage on a successful saving 
        throw or a missed attack. If your spell doesn't deal damage on a successful save, you can increase the damage 
        by 25 percent. \n
        
        You can use different damage dice than the ones in the table, provided that the average result is about the 
        same. Doing so can add a little variety to the spell. For example, you could change a cantrip's damage from 
        1d10 (average 5.5) to 2d4 (average 5), reducing the maximum damage and making an average result more likely. \n
        
        USE THIS TABLE TO DETERMINE SPELL DAMAGE: \n
        Spell Level One Target  Multiple Targets \n
        Cantrip	    1d10	    1d6 \n
        1st     	2d10	    2d6 \n
        2nd 	    3d10	    4d6 \n
        3rd     	5d10	    6d6 \n
        4th	        6d10	    7d6 \n
        5th	        8d10	    8d6 \n
        6th	        10d10   	11d6 \n
        7th     	11d10	    12d6 \n
        8th     	12d10	    13d6 \n
        9th	        15d10	    14d6 \n
        
        Healing Spells: \n
        You can also use the Spell Damage table to determine how many hit points a healing spell restores.  \n
        A cantrip shouldn't offer healing. \n
        
        Balance: \n
        Make sure that the spells power matches its level. \n   
        Spells should be balanced against other spells of the same level. \n
        Cantrips should only have one weak effect. \n
        
        Concentration: \n
        Spells with concentration have an effect every turn spend concentrating. \n 
        """

    level_for_request = selected_level
    school_for_request = selected_school
    range_for_request = selected_range
    components_for_request = selected_components
    spell_type_for_request = selected_spell_type
    save_type_for_request = selected_save_type
    spell_area_for_request = selected_spell_area
    requires_concentration_for_request = selected_requires_concentration
    damage_type_for_request = selected_damage_type
    condition_for_request = selected_condition
    ritual_for_request = selected_ritual
    casting_time_for_request = selected_casting_time
    spell_duration_for_request = selected_spell_duration

    if selected_level == "Random":
        level_for_request = random.choice(possible_spell_levels)
    if selected_school == "Random":
        school_for_request = random.choice(possible_spell_schools)
    if selected_range == "Random":
        range_for_request = ""
    if selected_components == "Random":
        components_for_request = ""
    if selected_spell_type == "Random":
        spell_type_for_request = ""
    if selected_save_type == "Random":
        save_type_for_request = ""
    if selected_spell_area == "Random":
        spell_area_for_request = ""
    if selected_requires_concentration == "Random":
        requires_concentration_for_request = ""
    if selected_damage_type == "Random":
        damage_type_for_request = ""
    if selected_condition == "Random":
        condition_for_request = ""
    if selected_ritual == "Random":
        ritual_for_request = random.choice(["Yes", "No"])
    if selected_casting_time == "Random":
        casting_time_for_request = ""
    if selected_spell_duration == "Random":
        spell_duration_for_request = ""


    prompt = f"""
        Create a Spell for Dungeons and Dragons. According to the previous format. Here is what I had in mind: \n
        Level: {level_for_request} \n
        School: {school_for_request} \n
        Range: {range_for_request} \n
        Casting Time: {casting_time_for_request} \n
        Duration: {spell_duration_for_request} \n
        Requires Concentration: {requires_concentration_for_request} \n
        Ritual: {ritual_for_request} \n
        Spell Type: {spell_type_for_request} \n
        Spell Components: {", ".join(components_for_request)} \n
        Save Type: {save_type_for_request} \n
        Spell Area: {spell_area_for_request} \n
        Damage Type: {damage_type_for_request} \n
        Condition: {condition_for_request} \n
        
        Description always has priority over the other attributes. \n
        Description: {description}
        """

    chatgpt_messages = [
        {"role": "system", "content": "Create a Spell for Dungeons and Dragons."},
        {"role": "system", "content": return_format},
        {"role": "system", "content": attribute_explanation},
        {"role": "system", "content": guide},
        {"role": "user", "content": prompt}
    ]

    return chatgpt_messages

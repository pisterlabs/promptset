# Scott Ratchford
# This file contains functions for generating campaign objects through API calls.

import openai
import datetime
import tiktoken
import campaign
import random
import regex as re
import json

# global constants for sanity checking
MAX_LOCATIONS = 10
MAX_CHARACTERS = 10
# global constants for World generation
RELATIONSHIP_PROBABILITY = 0.3
# global constants for generations
RETRY_LIMIT = 3         # number of times to retry a request before giving up
MODEL = "gpt-3.5-turbo" # AI model to use for generation
# global constants for logging
LOG_COMPLETIONS = False          # whether to log AI completions
SAVE_CREATIONS = True   # whether to save generated objects to files
# global constants for AI parameters
CREATIVE_TEMPERATURE = 1.3       # AI temperature

def remove_non_ascii(text):
    """Removes all non-ASCII characters from a string."""
    return re.sub(r'[^\x00-\x7F]', '', text)

def remove_double_newlines(text):
    """Removes all newline characters from a response."""
    text = re.sub(r' \n\n', '', text)
    return re.sub(r'\n\n', '', text)

def remove_incomplete_sentences(text):
    """Removes all incomplete sentences from a response."""
    return re.sub(r'\.([^\.]*)$', '', text) + "."

def remove_leading_whitespace(text):
    """Removes all leading whitespace from a response."""
    return re.sub(r'^\s+', '', text)

def remove_whitespace(text):
    """Removes all whitespace from a response."""
    return re.sub(r'\s+', '', text)

def make_printable(text):
    """Removes all non-printable characters from a string."""
    return remove_leading_whitespace(remove_incomplete_sentences(remove_double_newlines(remove_non_ascii(text))))

def create_and_log(completion: openai.ChatCompletion) -> None:
    """Accepts a ChatCompletion object and logs it to a file.

    Args:
        completion (openai.ChatCompletion): AI response to a user message.
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("./log/" + time + "-completion.json", "w") as f:
        f.write(str(completion))

def save_campaign_object(obj: campaign.World or campaign.Location or campaign.Character or campaign.Relationship or campaign.Item) -> None:
    """Saves a campaign object to a file.

    Args:
        obj (campaign.World or campaign.Location or campaign.Character or campaign.Relationship or campaign.Item): The object to save.
    """
    if(type(obj) == campaign.World):
        with open("./data/" + type(obj).__name__ + "-" + obj.name + ".json", "w") as f:
            f.write(json.dumps(obj, indent=4, cls=campaign.WorldEncoder, ensure_ascii=True))
        return
    elif(type(obj) == campaign.Location):
        with open("./data/" + type(obj).__name__ + "-" + obj.name + ".json", "w") as f:
            f.write(json.dumps(obj, indent=4, cls=campaign.LocationEncoder, ensure_ascii=True))
        return
    elif(type(obj) == campaign.Character):
        with open("./data/" + type(obj).__name__ + "-" + obj.name + ".json", "w") as f:
            f.write(json.dumps(obj, indent=4, cls=campaign.CharacterEncoder, ensure_ascii=True))
        return
    elif(type(obj) == campaign.Item):
        with open("./data/" + type(obj).__name__ + "-" + obj.name + ".json", "w") as f:
            f.write(json.dumps(obj, indent=4, cls=campaign.ItemEncoder, ensure_ascii=True))
        return
    elif(type(obj) == campaign.Relationship):
        with open("./data/" + type(obj).__name__ + "-" + obj.characterAName + "-" + obj.characterBName + ".json", "w") as f:
            f.write(json.dumps(obj, indent=4, cls=campaign.RelationshipEncoder, ensure_ascii=True))
        return
    else:
        with open("./data/" + type(obj).__name__ + "-" + obj.name + ".json", "w") as f:
            f.write(json.dumps(str(obj)))

def estimate_cost(prompt: str, modelName: str, returnType: type = float) -> float or str:
    """Estimates the cost of a prompt in USD. A very rough estimate.

    Args:
        prompt (str): Prompt to estimate cost of.
        modelName (str): Model to use for cost estimation.
        returnType (type, optional): Type to return. Defaults to float.
    """
    # per 1000 tokens
    # TODO: add more models and costs
    model_costs = {
        "gpt-3.5-turbo": 0.002
    }
    numTokens = len(tiktoken.Encoding.encode(prompt))
    if(returnType == str):
        return "$" + str(numTokens / 1000 * model_costs[modelName])
    else:
        return numTokens / 1000 * model_costs[modelName]

def generate_location(world: campaign.World) -> campaign.Location:
    """Generates a location using the OpenAI API.

    Args:
        world (campaign.World): The world to add the location to.

    Raises:
        ValueError: API response was too long.
        ValueError: API response was invalid.

    Returns:
        campaign.Location: The generated location.
    """
    all_messages = world.locations_as_system_msg()  # add the world as context for the next location
    all_messages.append(
        {"role": "user", "content": "Generate a location to add to the world of " + str(world.name) + ". Create your reply in the format: location_name|location_description. Use only printable ASCII characters. Do not use the _ character."}
    )
    reply = openai.ChatCompletion.create(model=MODEL, messages=all_messages, temperature=CREATIVE_TEMPERATURE, max_tokens=300)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("Location generation failed due to length.")
    try:
        message = reply.choices[0].message.content.split("|")
        location = campaign.Location(message[0])
        location.description = message[1]
    except:
        raise ValueError("Location generation failed due to invalid response. The | character was not used.")
    if(SAVE_CREATIONS):
        save_campaign_object(location)
    return location

def generate_character(world: campaign.World) -> campaign.Character:
    """Generates a character using the OpenAI API.

    Args:
        world (campaign.World): The world to add the character to.

    Raises:
        ValueError: API response was too long.
        ValueError: API response was invalid.

    Returns:
        campaign.Character: The generated character.
    """
    all_messages = [world.as_system_msg()]  # add the world as context for the next character, TODO: use a shorter context
    all_messages.append(
        {"role": "user", "content": "Generate a character to add to the world of " + str(world.name) + ". Create your reply in the format: character_name|character_description. Use only printable ASCII characters. Do not use the _ character."}
    )
    reply = openai.ChatCompletion.create(model=MODEL, messages=all_messages, temperature=CREATIVE_TEMPERATURE, max_tokens=300)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("Character generation failed due to length.")
    try:
        message = reply.choices[0].message.content.split("|")
        character = campaign.Character(message[0])
        character.description = message[1]
    except:
        raise ValueError("Character generation failed due to invalid response. The | character was not used.")
    if(SAVE_CREATIONS):
        save_campaign_object(character)
    return character

def generate_symmetric_relationship(characterA: campaign.Character, characterB: campaign.Character) -> campaign.Relationship:
    """Generates a symmetric relationship between two characters using the OpenAI API.

    Args:
        characterA (campaign.Character): The first character.
        characterB (campaign.Character): The second character.
    
    Raises:
        ValueError: If the relationship generation fails due to length or invalid response.

    Returns:
        campaign.Relationship: The generated relationship.
    """
    all_messages = [characterA.as_system_msg(), characterB.as_system_msg()]  # add the characters as context for the next relationship
    all_messages.append(
        {"role": "user", "content": "Generate a relationship between " + str(characterA.name) + " and " + str(characterB.name) + ". Create your reply in the format: relationship_description. Use only printable ASCII characters. Do not use the _ character."}
    )
    reply = openai.ChatCompletion.create(model=MODEL, messages=all_messages, temperature=CREATIVE_TEMPERATURE, max_tokens=300)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("Relationship generation failed due to length.")
    try:
        message = reply.choices[0].message.content
        relationship = campaign.Relationship(characterA, characterB)
        relationship.set_symmetric_relationship(message)
    except:
        raise ValueError("Relationship generation failed due to invalid response.")
    if(SAVE_CREATIONS):
        save_campaign_object(relationship)
    return relationship

def generate_asymmetric_relationship(characterA: campaign.Character, characterB: campaign.Character) -> campaign.Relationship:
    """Generates an asymmetric relationship between two characters using the OpenAI API.

    Args:
        characterA (campaign.Character): The first character.
        characterB (campaign.Character): The second character.

    Raises:
        ValueError: If the relationship generation fails due to length or invalid response.

    Returns:
        campaign.Relationship: The generated relationship.
    """
    all_messages = [characterA.as_system_msg(), characterB.as_system_msg()]  # add the characters as context for the next relationship
    all_messages.append(
        {"role": "user", "content": "Generate a relationship between " + str(characterA.name) + " and " + str(characterB.name) + " that is the same in both directions. Create your reply in the format: relationship_description. Use only printable ASCII characters. Do not use the _ character. Limit your response to 30 words."}
    )
    reply = openai.ChatCompletion.create(model=MODEL, messages=all_messages, temperature=CREATIVE_TEMPERATURE, max_tokens=300)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("Relationship generation failed due to length.")
    try:
        message = reply.choices[0].message.content.split("|")
        relationship = campaign.Relationship(characterA, characterB)
        relationship.set_asymmetric_relationship(message[0], message[1])
    except:
        raise ValueError("Relationship generation failed due to invalid response. The | character was not used.")
    if(SAVE_CREATIONS):
        save_campaign_object(relationship)
    return relationship

def generate_item(world_basics: str, location: campaign.Location) -> campaign.Item:
    """Generates an item using the OpenAI API.

    Args:
        world_description (str): The description of the world the item is in.

    Raises:
        ValueError: If the item generation fails due to length or invalid response.

    Returns:
        campaign.Item: The generated item.
    """
    all_messages = [{"role": "system", "content": world_basics + str(location)}]  # add the world as context for the next item
    all_messages.append(
        {"role": "user", "content": "Generate an item to add to the world. Create your reply in the format: item_name|item_description. Use only printable ASCII characters. Do not use the _ character."}
    )
    reply = openai.ChatCompletion.create(model=MODEL, messages=all_messages, temperature=CREATIVE_TEMPERATURE, max_tokens=300)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("Item generation failed due to length.")
    try:
        message = reply.choices[0].message.content.split("|")
        item = campaign.Item(message[0])
        item.description = message[1]
    except:
        raise ValueError("Item generation failed due to invalid response. The | character was not used.")
    if(SAVE_CREATIONS):
        save_campaign_object(item)
    return item

def generate_world(numLocations: int = 0, numCharacters: int = 0, numItems: int = 0) -> campaign.World:
    """Generates a world using the OpenAI API.

    Args:
        numLocations (int, optional): The number of Locations to generate. Defaults to 0.
        numCharacters (int, optional): The number of Characters to generate. Defaults to 0.
        numItems (int, optional): The number of Items to generate. Defaults to 0.

    Returns:
        campaign_classes.World: The generated world.
    """

    # sanity checks to prevent excessive API calls
    if(numLocations < 0):
        raise ValueError("numLocations must be greater than or equal to 0.")
    if(numLocations > MAX_LOCATIONS):
        raise ValueError("numLocations must be less than or equal to " + str(MAX_LOCATIONS) + ".")
    if(numLocations < 0):
        raise ValueError("numCharacters must be greater than or equal to 0.")
    if(numLocations > MAX_CHARACTERS):
        raise ValueError("numCharacters must be less than or equal to " + str(MAX_CHARACTERS) + ".")

    world_prompt = [
        {"role": "user", "content": "Generate a world for a 5e campaign."},
        {"role": "system", "content": "Give your reply in the format: world_name|world_description"},
        {"role": "system", "content": "Use only printable ASCII characters. Do not use the _ character."},
    ]
    reply = openai.ChatCompletion.create(model=MODEL, messages=world_prompt, temperature=CREATIVE_TEMPERATURE, max_tokens=500)
    if(LOG_COMPLETIONS):
        create_and_log(reply)
    if(reply.choices[0].finish_reason == "length"):
        raise ValueError("World generation failed due to length. Try again.")
    try:
        message = reply.choices[0].message.content.split("|")
        world = campaign.World(remove_whitespace(message[0]), message[1])
    except:
        raise ValueError("World generation failed due to invalid response. Try again.")

    for i in range(numLocations):
        for j in range(RETRY_LIMIT):
            try:
                location = generate_location(world)
                break
            except ValueError as e:
                continue
        if(location in locals()):
            continue
        world.add_location(location)
    
    for i in range(numCharacters):
        for j in range(RETRY_LIMIT):
            try:
                character = generate_character(world)
                break
            except ValueError as e:
                continue
        if(character in locals()):
            continue
        world.add_character(character)
    
    for characterA in world.characters:
        for characterB in world.characters:
            if(characterA == characterB):   # don't generate relationships between the same character
                continue
            if(world.get_relationship_between(characterA, characterB) != None):   # don't generate relationships between characters that already have a relationship
                continue
            if(random.random() < RELATIONSHIP_PROBABILITY):   # generate a relationship between the two characters only if the random number is less than the probability
                if(random.random() < 0.25):
                    # asymmetric relationship
                    for j in range(RETRY_LIMIT):
                        try:
                            relationship = generate_asymmetric_relationship(characterA, characterB)
                            break
                        except ValueError as e:
                            continue
                else:
                    # symmetric relationship
                    for j in range(RETRY_LIMIT):
                        try:
                            relationship = generate_symmetric_relationship(characterA, characterB)
                            break
                        except ValueError as e:
                            continue
                if(relationship in locals()):
                    continue
                world.add_relationship(relationship)    # add the relationship to the world

    for i in range(numItems):
        location = random.choice(world.locations)
        for j in range(RETRY_LIMIT):
            try:
                item = generate_item(world.world_basics(), location)
                break
            except ValueError as e:
                continue
        if(item in locals()):
            continue    # if the item generation failed, skip this item
        location.add_item(item)

    if(SAVE_CREATIONS):
        save_campaign_object(world)
    return world

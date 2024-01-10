import json
import os
import openai
import requests
import logging
from PIL import Image

from . import elements
from . import chat_functions


IMAGE_DIRECTORY="/tmp"
TESTING=False


STATE_WORLDS = "State_Worlds"
STATE_WORLD = "State_World"
STATE_WORLD_EDIT = "State_World_Edit"
STATE_CHARACTERS = "State_Characters"
STATE_ITEMS = "State_Items"
STATE_SITES = "State_Sites"

def elemTypeToState(element_type):
  if element_type == elements.ElementType.WorldType():
    return STATE_WORLD
  elif element_type == elements.ElementType.CharacterType():
    return STATE_CHARACTERS    
  elif element_type == elements.ElementType.ItemType():
    return STATE_ITEMS    
  elif element_type == elements.ElementType.SiteType():
    return STATE_SITES
  return STATE_WORLDS


states = {
  STATE_WORLDS: [ "ListWorlds", "ReadWorld", "CreateWorld" ],
  STATE_WORLD: [ "ReadWorld", "ReadPlanningNotes", 
                 "ReadCharacter", "ReadItem", "ReadSite",
                 "ChangeState", "EditWorld" ],
  STATE_WORLD_EDIT: [ "UpdateWorld", "ReadWorld",
                      "ReadPlanningNotes", "UpdatePlanningNotes",
                      "CreateWorldImage", "ChangeState" ],
  STATE_CHARACTERS: [ "ListCharacters", "ReadCharacter",
                      "CreateCharacter", "UpdateCharacter",
                      "ReadPlanningNotes", 
                      "CreateCharacterImage", "ChangeState" ],
  STATE_ITEMS: [ "ListItems", "ReadItem",
                 "CreateItem", "UpdateItem",
                 "ReadPlanningNotes",                  
                 "CreateItemImage", "ChangeState" ],
  STATE_SITES: [ "ListSites", "ReadSite",
                 "CreateSite", "UpdateSite",
                 "ReadPlanningNotes",                  
                 "CreateSiteImage", "ChangeState" ],
}



  
GLOBAL_INSTRUCTIONS = """
You are a co-designer of fictional worlds, developing ideas
and and backstories for these worlds and the contents of worlds, including
new unique fictional characters. Create new characters, don't use existing characters.

We design the world with a name and a high level description and create background details

We use Planning Notes for plans on characters, items, and sites.

We can be in one of the following states:
- State_Worlds: We can open existing worlds and create new worlds
- State_World: We view a world description, details, and PlanningNotes.
- State_World_Edit: We change a world description, details, and PlanningNotes.
- State_Characters: We can view characters and create new characters and change the description and details of a character.
- State_Items: We can view items and create new items and change the description and details of an item.
- State_Sites: We can view sites and create new sites and change the description and details of a site.

The current state is "{current_state}"

Suggest good ideas for descriptions and details for the worlds, characters, sites, and items.
Suggest next steps to the user in designing a complete world.

"""
  

instructions = {
  STATE_WORLDS:
"""
You can create a new world or resume work on an existing one by reading it.
To modify an existing world, ChangeState to State_World.
To get a list of worlds, call ListWorlds
Get a list of worlds before reading a world or creating a new one
Before creating a new world, check if it already exists by using ListWorlds
""",

  STATE_WORLD:
  """
We are working on the world "{current_world_name}"

A world has plans that list the planned main characters, key sites, and special items. Read plans for the world by calling ReadPlanningNotes  

Modify world attributes by calling EditWorld

To view, create, or update characters, change state to State_Characters.
To view, create, or update items, change state to State_Items.
To view, create, or update sites, change state to State_Sites.  

  """,

  STATE_WORLD_EDIT:
  """
We are working on the world "{current_world_name}"
  
A world needs a short high level description refelcting the nature of the world.

A world has details, that give more information about the world such as the backstory.

A world has plans that list the planned main characters, key sites, and special items. Read plans for the world by calling ReadPlanningNotes, update the plans with UpdatePlanningNotes.

Build prompts to create images using information from the description and details in the prompt.

Save information about the world by calling UpdateWorld

To view information about characters, items, or sites, change the state to State_World
  """,
  
  STATE_CHARACTERS:
"""
We are working on world "{current_world_name}"

Worlds have charaters which are actors in the world with a backstory, abilities, and motivations.  You can create characters and change information about the characters.

You can update the name, description, and details of the character.
You save changes to a character by calling UpdateCharacter.  

Use information in the world details to guide character creation and design.

Before creating a new character, check if it already exists by calling the ListCharacters function.

When creating images for the character using CreateCharacterImage, make a long prompt using the character description and details.

Save detailed information about the character in character details.

To work on information about the world call ChangeState
To work on items or sites, call ChangeState
""",

  STATE_ITEMS:
"""
We are working on world "{current_world_name}"

Worlds have items which exist in the world and have special significance.  You can create items and change information about the items.

You can update the name, description, and details of an item.
You save changes to an item by calling UpdateItem.  

Use information in the world details to guide item creation and design.

Before creating a new item, check if it already exists by calling the ListItems function.

When creating images for the item with CreateItemImage, make a long prompt using the item description and details.

Save detailed information about the item in item details.

To view or change information about the world call ChangeState
To view or work characters or sites, call ChangeState
""",

  STATE_SITES:
"""
We are working on world "{current_world_name}"

Worlds have sites which are significant locations. Cities, buildings, and special areas may all be sites. You can create sites and change information about the sites.

You can update the name, description, and details of a site.
You save changes to a site by calling UpdateSite.  

Use information in the world details to guide site creation and design.

Before creating a new site, check if it already exists by calling the ListSites function.

When creating images for the site with CreateSiteImage, make a long prompt using the site description and details.


Save detailed information about the site in site details.

To work on information about the world call ChangeState
To work on characters or items, call ChangeState
""",

}


def checkDuplication(name, element_list):
  """
  Check for any collisions between name and existing list
  element_list: list returned from getElements

  Return None if no conflict
  Return an id if a conflict

  """
  # Check if name is a substring of any existing name
  name = name.lower()
  for element in element_list:
    if name in element.getName().lower():
      return element.getID()

  # Check if any existing name is a substring of the new name
  for element in element_list:
    if element.getName().lower() in name:
      return element.getID()

  return None


class DesignFunctions(chat_functions.BaseChatFunctions):

  def __init__(self):
    chat_functions.BaseChatFunctions.__init__(self)
    self.current_state = STATE_WORLDS
    self.current_world_name = None

    # Tracks current world, current element
    self.current_view = elements.ElemTag()
    
    # An ElemTag that describes a view we need to change into.
    # This happens when the user changes the view in the UI.
    # We need to sync the GPT to the new view
    self.next_view = elements.ElemTag()


  def getCurrentWorldID(self):
    # May return None
    return self.current_view.getWorldID()
  
  def get_instructions(self, db):
    global_instructions = GLOBAL_INSTRUCTIONS.format(
      current_state=self.current_state)
    return global_instructions + "\n" + self.get_state_instructions()
  
  def get_state_instructions(self):
    value = instructions[self.current_state].format(
      current_world_name = self.current_world_name)
    return value

  def get_available_tools(self):
    return self.get_available_tools_for_state(self.current_state)

  def get_available_tools_for_state(self, state):
    functions = {}
    for function in all_functions:
      functions[function["name"]] = function

    result = []
    for name in states[state]:
      tool = { "type": "function",
               "function": functions[name] }   
      result.append(tool)
    return result

  def track_tokens(self, db, prompt, complete, total):
    world_id = self.getCurrentWorldID()
    if world_id is None:
      world_id = 0
    chat_functions.track_tokens(db, world_id, prompt, complete, total)

  def get_view(self):
    return self.current_view.json()

  def set_view(self, next_view):
    """
    Set the target view.
    If same as current, this is a NO-OP
    """
    next_view = elements.ElemTag.JsonTag(next_view)
    if next_view == self.current_view:
      # View already matches - reset
      self.next_view = elements.ElemTag()
      return
    
    self.next_view = next_view


  def checkToolChoice(self, history):
    """
    Determine if we need to fetch additional information
    to act on requests.

    Use the current state and the presense of included messages
    to make decisions.
    """
    tool_func = None

    logging.info(f"state: {self.current_state}")
    logging.info("current view: %s",
                 self.current_view.jsonStr())
    logging.info("next view: %s",
                 self.next_view.jsonStr())
      
    # Check if the proper list is loaded for the current state.
    if self.current_state == STATE_WORLDS:
      if not history.hasToolCall("ListWorlds", {}):
        tool_func = "ListWorlds"
    elif self.current_state == STATE_CHARACTERS:
      if not history.hasToolCall("ListCharacters", {}):
        tool_func = "ListCharacters"
    elif self.current_state == STATE_ITEMS:
      if not history.hasToolCall("ListItems", {}):
        tool_func = "ListItems"
    elif self.current_state == STATE_SITES:
      if not history.hasToolCall("ListSites", {}):
        tool_func = "ListSites"

    if tool_func is not None:
      return { "type": "function",
               "function": { "name": tool_func }}
    return None
    
  def execute_function_call(self, db, function_name, arguments):
    """
    Dispatch function for function_name
    Takes:
      function_name - string
      arguments - dict build from json.loads
    Returns
      dict ready for json.dumps
    """
    # Default response value
    result = '{ "error": "' + f"no such function: {function_name}" + '" }'

    if function_name not in states[self.current_state]:
      result = self.funcError(f"No available function {function_name}. " +
                              "Perhaps call ChangeState")

    elif function_name == "EditWorld":
      self.current_state = STATE_WORLD_EDIT
      result = self.funcStatus("edit enabled")

    elif function_name == "ChangeState":
      result = self.FuncChangeState(db, arguments)

    elif function_name == "CreateWorld":
      result = self.FuncCreateWorld(db, arguments)

    elif function_name == "ListWorlds":
      result = [ { "id": entry.getID(), "name": entry.getName() }
                   for entry in elements.listWorlds(db) ]
      
    elif function_name == "UpdateWorld":
      result = self.FuncUpdateWorld(db, arguments)

    elif function_name == "ReadWorld":
      result = self.FuncReadWorld(db, arguments)

    elif function_name == "ReadPlanningNotes":
      result = self.FuncReadPlanningNotes(db, arguments)

    elif function_name == "UpdatePlanningNotes":
      result = self.FuncUpdatePlanningNotes(db, arguments)

    elif function_name == "ListCharacters":
      result = [{ "id": entry.getID(), "name": entry.getName() }            
           for entry in elements.listCharacters(db, self.getCurrentWorldID())]

    elif function_name == "ReadCharacter":
      result = self.FuncReadCharacter(db, arguments)
  
    elif function_name == "CreateCharacter":
      result = self.FuncCreateCharacter(db, arguments)

    elif function_name == "UpdateCharacter":
      result = self.FuncUpdateCharacter(db, arguments)

    elif function_name == "ListItems":
      result = [ { "id": entry.getID(), "name": entry.getName() } 
             for entry in elements.listItems(db, self.getCurrentWorldID()) ]

    elif function_name == "ReadItem":
      result = self.FuncReadItem(db, arguments)
  
    elif function_name == "CreateItem":
      result = self.FuncCreateItem(db, arguments)

    elif function_name == "UpdateItem":
      result = self.FuncUpdateItem(db, arguments)
      
    elif function_name == "ListSites":
      result = [ { "id": entry.getID(), "name": entry.getName() } 
              for entry in elements.listSites(db, self.getCurrentWorldID()) ]

    elif function_name == "ReadSite":
      result = self.FuncReadSite(db, arguments)
  
    elif function_name == "CreateSite":
      result = self.FuncCreateSite(db, arguments)

    elif function_name == "UpdateSite":
      result = self.FuncUpdateSite(db, arguments)
      
    elif (function_name == "CreateWorldImage" or
          function_name == "CreateCharacterImage" or
          function_name == "CreateItemImage" or
          function_name == "CreateSiteImage"):          
      result = self.FuncCreateImage(db, arguments)

    if self.current_state == STATE_WORLDS:
      self.current_view = elements.ElemTag()
      self.current_world_name = None
    elif self.current_state == STATE_WORLD:
      self.current_view = elements.ElemTag.WorldTag(self.getCurrentWorldID())
    return result

  def FuncChangeState(self, db, arguments):
    state = arguments["state"]
    if states.get(state) is None:
      return self.funcError(f"unknown state: {state}")

    # Check is state is legal
    if ((state == STATE_WORLD or
         state == STATE_CHARACTERS) and self.current_view.noElement()):
      return self.funcError(f"Must read or create a world for {state}")
    self.current_state = state

          
    return self.funcStatus(f"state changed: {state}")

  def FuncCreateWorld(self, db, arguments):
    world = elements.World()
    world.setName(arguments["name"])
    world.updateProperties(arguments)

    # Check for duplicates
    worlds = elements.listWorlds(db)    
    name = checkDuplication(world.getName(), worlds)
    if name is not None:
      return self.funcError(f"Similar name already exists: {name}")

    world = elements.createWorld(db, world)
    self.current_state = STATE_WORLD
    self.current_view = world.getElemTag()
    self.current_world_name = world.getName()
    self.modified = True      
    status = self.funcStatus("created world")
    status["id"] = world.id
    return status

  def FuncUpdateWorld(self, db, arguments):
    world = elements.loadWorld(db, self.getCurrentWorldID())
    if world is None:
      return self.funcError(f"World not found {self.getCurrentWorldID()}")
    world.updateProperties(arguments)
    # TODO: check name collision    
    elements.updateWorld(db, world)
    self.modified = True
    status = self.funcStatus("updated world")    
    status["id"] = world.id
    return status

  def FuncReadWorld(self, db, arguments):
    id = arguments["id"]
    world = elements.loadWorld(db, id)
    if world is None:
      return self.funcError(f"no world '{id}'")      
    content = { "id": world.id,
                **world.getProperties(),
                "has_image": world.hasImage(), 
               }
    # Don't include plans in the world description
    if elements.PROP_PLANS in content.keys():
      del content[elements.PROP_PLANS]

    # Add information on the existing elements of the world.
    content["has_plans"] = len(world.getPlans()) > 0
    
    population = []
    population.append("Existing Characters:\n")
    for character in elements.listCharacters(db, world.id):
      population.append(f"- {character.getID()}: {character.getName()}")
    population.append("")

    population.append("Existing Items:\n")        
    for item in elements.listItems(db, world.id):
      population.append(f"- {item.getID()}: {item.getName()}")
    population.append("")
    
    population.append("Existing Sites:\n")        
    for site in elements.listSites(db, world.id):
      population.append(f"- {site.getID()}: {site.getName()}")

    content["elements"] = "\n".join(population)

    # Side affect, change state
    self.current_state = STATE_WORLD
    self.current_view = world.getElemTag()
    self.current_world_name = world.getName()
      
    return content

  def FuncReadPlanningNotes(self, db, arguments):
    world = elements.loadWorld(db, self.getCurrentWorldID())
    if world is None:
      return self.funcError(f"World not found {self.getCurrentWorldID()}")
    content = { "id": world.id,
               elements.PROP_PLANS: world.getPlans() }

    # Side affect, change state
    # TODO: this will not change here - remove it
    self.current_state = STATE_WORLD
    self.current_view = world.getElemTag()
    self.current_world_name = world.getName()
    return content

  def FuncUpdatePlanningNotes(self, db, arguments):
    world = elements.loadWorld(db, self.getCurrentWorldID())
    if world is None:
      return self.funcError(f"World not found {self.getCurrentWorldID()}")
    world.setPlans(arguments[elements.PROP_PLANS])
    elements.updateWorld(db, world)
    self.modified = True
    status = self.funcStatus("updated world plans")    
    status["id"] = world.id
    return status
  
  def FuncReadCharacter(self, db, arguments):
    id = arguments.get("id")
    if id is None:
      return self.funcError("request missing id parameter")
    
    character = elements.loadCharacter(db, id)
    if character is not None:
      content = { "id": character.id,
                  **character.getProperties(),
                  "has_image": character.hasImage(),                   
                 }
      self.current_state = STATE_CHARACTERS
      self.current_view  = character.getElemTag()
    else:
      return self.funcError(f"no character '{id}'")
    return content
  
  def FuncCreateCharacter(self, db, arguments):
    character = elements.Character(self.getCurrentWorldID())
    character.setName(arguments["name"])

    characters = elements.listCharacters(db, self.getCurrentWorldID())    
    name = checkDuplication(character.getName(), characters)
    if name is not None:
      return self.funcError(f"Similar name already exists: {name}")

    character.updateProperties(arguments)    
    character = elements.createCharacter(db, character)
    self.current_view  = character.getElemTag()
    self.current_state = STATE_CHARACTERS   
    status = self.funcStatus("Created character")
    status["id"] = character.id
    return status
    
  def FuncUpdateCharacter(self, db, arguments):
    id = arguments["id"]
    character = elements.loadCharacter(db, id)
    if character is None:
      return self.funcError(f"Character not found {id}")
    character.updateProperties(arguments)
    # TODO: check name collision
    elements.updateCharacter(db, character)
    self.modified = True
    self.current_view  = character.getElemTag()
    status = self.funcStatus("Updated character")
    status["id"] = id
    return status

  def FuncReadItem(self, db, arguments):
    id = arguments.get("id")
    if id is None:
      return self.funcError("request missing id parameter")
    
    item = elements.loadItem(db, id)
    if item is not None:
      content = { "id": item.id,
                  **item.getProperties(),
                  "has_image": item.hasImage(),                  
                 }
      self.current_state = STATE_ITEMS
      self.current_view = item.getElemTag()
    else:
      return self.funcError(f"no item '{id}'")
    return content
  
  def FuncCreateItem(self, db, arguments):
    item = elements.Item(self.getCurrentWorldID())
    item.setName(arguments["name"])

    items = elements.listItems(db, self.getCurrentWorldID())    
    name = checkDuplication(item.getName(), items)
    if name is not None:
      return self.funcError(f"Similar name already exists: {name}")

    item.updateProperties(arguments)    
    item = elements.createItem(db, item)
    self.current_view  = item.getElemTag()
    self.current_state = STATE_ITEMS   
    status = self.funcStatus("Created item")
    status["id"] = item.id
    return status
    
  def FuncUpdateItem(self, db, arguments):
    id = arguments["id"]
    item = elements.loadItem(db, id)
    if item is None:
      return self.funcError(f"Item not found {id}")
    item.updateProperties(arguments)
    # TODO: check name collision
    elements.updateItem(db, item)
    self.modified = True
    self.current_view  = item.getElemTag()
    status = self.funcStatus("Updated item")
    status["id"] = id
    return status

  def FuncReadSite(self, db, arguments):
    id = arguments.get("id")
    if id is None:
      return self.funcError("request missing id parameter")
    
    site = elements.loadSite(db, id)
    if site is not None:
      content = { "id": site.id,
                  **site.getProperties(),
                  "has_image": site.hasImage(),
                 }
      self.current_state = STATE_SITES
      self.current_view  = site.getElemTag()
    else:
      return self.funcError(f"no site '{id}'")
    return content
  
  def FuncCreateSite(self, db, arguments):
    site = elements.Site(self.getCurrentWorldID())
    site.setName(arguments["name"])

    sites = elements.listSites(db, self.getCurrentWorldID())    
    name = checkDuplication(site.getName(), sites)
    if name is not None:
      return self.funcError(f"Similar name already exists: {name}")

    site.updateProperties(arguments)    
    site = elements.createSite(db, site)
    self.current_view  = site.getElemTag()
    self.current_state = STATE_SITES   
    status = self.funcStatus("Created site")
    status["id"] = site.id
    return status
    
  def FuncUpdateSite(self, db, arguments):
    id = arguments["id"]
    site = elements.loadSite(db, id)
    if site is None:
      return self.funcError(f"Site not found {id}")
    site.updateProperties(arguments)
    # TODO: check name collision
    elements.updateSite(db, site)
    self.current_view  = site.getElemTag()
    self.modified = True      
    status = self.funcStatus("Updated site")
    status["id"] = id
    return status

  def FuncCreateImage(self, db, arguments):
    # Check if the budget allows
    if not chat_functions.check_image_budget(db):
      return self.funcError("No budget available for image creation")
    
    image = elements.Image()
    image.setPrompt(arguments["prompt"])
    logging.info("Create image: prompt %s", image.prompt)
    if self.current_state == STATE_CHARACTERS:
      id = arguments["id"]
      character = elements.loadCharacter(db, id)
      if character is None:
        return self.funcError(f"no character '{id}'")
        
      image.setParentId(id)
      self.current_view = character.getElemTag()
    elif self.current_state == STATE_ITEMS:
      id = arguments["id"]
      item = elements.loadItem(db, id)
      if item is None:
        return self.funcError(f"no item '{id}'")
        
      image.setParentId(id)
      self.current_view = item.getElemTag()
      
    elif self.current_state == STATE_SITES:
      id = arguments["id"]
      site = elements.loadSite(db, id)
      if site is None:
        return self.funcError(f"no site '{id}'")
        
      image.setParentId(id)
      self.current_view = site.getElemTag()

    else:
      image.setParentId(self.getCurrentWorldID())

    if image.parent_id is None:
      logging.info("create image error: empty parent_id")
      return self.funcError("internal error - no id")

    dest_file = os.path.join(IMAGE_DIRECTORY, image.getFilename())
    logging.info("dest file: %s", dest_file)
    result = image_get_request(
      "Produce a visual image that captures the following: " +
      image.prompt,
      dest_file)
    
    if result:
      logging.info("file create done, create image record")
      chat_functions.count_image(db, self.getCurrentWorldID(), 1)
      image = elements.createImage(db, image)
      create_image_thumbnail(image)
      self.modified = True
      status = self.funcStatus("created image")
      status["id"] = image.id
      return status
    return self.funcError("problem generating image")
  
def create_image_thumbnail(image_element):
  """
  Take an image element and create a thumbnail in the
  IMAGE_DIRECTORY
  """
  in_file = os.path.join(IMAGE_DIRECTORY, image_element.getFilename())
  out_file = os.path.join(IMAGE_DIRECTORY, image_element.getThumbName())
  image = Image.open(in_file)
  MAX_SIZE=(100, 100)
  image.thumbnail(MAX_SIZE)
  image.save(out_file)
  
  
    
def image_get_request(prompt, dest_file):
  # Testing stub. Just copy existing file.
  if TESTING:
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, "static/logo.png")
    with open(dest_file, "wb") as fout:
      with open(path, "rb") as fin:
        fout.write(fin.read())
    return True

  # Functional code. Generate image and copy to dest_file.
  headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + openai.api_key,
  }
  json_data = {"model": "dall-e-3",
               "size" : "1024x1024",
               "prompt": prompt }
  try:
    logging.info("post: %s", prompt)

    response = requests.post(
      "https://api.openai.com/v1/images/generations",
      headers=headers,
      json=json_data,
      timeout=60,
    )
    result = response.json()
    logging.info("image complete")
    if result.get("data") is None:
      return False
    
    response = requests.get(result["data"][0]["url"], stream=True)
    if response.status_code != 200:
      return False

    with open(dest_file, "wb") as f:
      response.raw.decode_content = True
      # Probably uses more memory than necessary
      # TODO: make more efficient
      f.write(response.raw.read())
    return True
      
  except Exception as e:
    logging.info("Unable to generate ChatCompletion response")
    logging.info("Exception: ", str(e))
    raise e



all_functions = [
  {
    "name": "EditWorld",
    "description": "Enable editing of world properties.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },
  {
    "name": "ChangeState",
    "description": "Change the current state for a new activity.",
    "parameters": {
      "type": "object",
      "properties": {
        "state": {
          "type": "string",
          "description": "The new state",
        },
      },
      "required": [ "state" ]            
    },
  },
  
  {
    "name": "ListWorlds",
    "description": "Get a list of existing worlds.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },
  
  {
    "name": "CreateWorld",
    "description": "Create a new virtual world",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the virtual world",
        },
        "description": {
          "type": "string",
          "description": "Short high level description of the virtual world",
        },
      },
      "required": [ "name", "description" ]      
    },
  },


  {
    "name": "ReadWorld",
    "description": "Read in a specific virtual world.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for world intance.",
        },
      },
      "required": [ "id"]
    },
  },
  
  
  {
    "name": "UpdateWorld",
    "description": "Update the values of the virtual world.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the virtual world.",
        },
        "description": {
          "type": "string",
          "description": "Short high level description of the world.",
        },
        "details": {
          "type": "string",
          "description": "Detailed information about the virtual world.",
        },
        "plans": {
          "type": "string",
          "description": "Plans for developing characters, items, and sites.",
        },
      },
    },
  },

  {
    "name": "CreateWorldImage",
    "description": "Create an image for the current world",
    "parameters": {
      "type": "object",
      "properties": {
        "prompt": {
          "type": "string",
          "description": "A prompt from which to create the image.",
        },
      },
      "required": [ "prompt" ],
    },
  },

  {
    "name": "ReadPlanningNotes",
    "description": "Read in the plans specific virtual world.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },

  {
    "name": "UpdatePlanningNotes",
    "description": "Update the plans of the virtual world.",
    "parameters": {
      "type": "object",
      "properties": {
        "plans": {
          "type": "string",
          "description": "Plans for the virtual world.",
        },
      },
    },
  },
  
  {
    "name": "ListCharacters",
    "description": "Get a characters in the current world.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },

  {
    "name": "CreateCharacter",
    "description": "Create a new character instance",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the character",
        },
        "description": {
          "type": "string",
          "description": "Short description of the character",
        },
      },
      "required": [ "name", "description" ]
    },
  },

  {
    "name": "ReadCharacter",
    "description": "Read in a specific character.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the character.",
        },
      },
      "required": [ "id"]
    },
  },

  {
    "name": "UpdateCharacter",
    "description": "Update the values of the character.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the character.",
        },
        "name": {
          "type": "string",
          "description": "Name of the character.",
        },
        "description": {
          "type": "string",
          "description": "Short description of the character",
        },
        "details": {
          "type": "string",
          "description": "Detailed information about the character.",
        },
        "personality": {
          "type": "string",
          "description": "Describes the personality of the character.",
        },
      },
      "required": [ "id"]      
    }
  },
  
  {
    "name": "CreateCharacterImage",
    "description": "Create an image for a specific character",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the character.",
        },
        "prompt": {
          "type": "string",
          "description": "A prompt from which to create the image.",
        },
      },
      "required": [ "id", "prompt" ],
    },
  },

  {
    "name": "ListItems",
    "description": "Get a items in the current world.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },

  {
    "name": "CreateItem",
    "description": "Create a new item instance",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the item",
        },
        "description": {
          "type": "string",
          "description": "Short description of the item",
        },
      },
      "required": [ "name", "description" ]
    },
  },

  {
    "name": "ReadItem",
    "description": "Read in a specific item.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the item.",
        },
      },
      "required": [ "id"]
    },
  },

  {
    "name": "UpdateItem",
    "description": "Update the values of the item.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the item.",
        },
        "name": {
          "type": "string",
          "description": "Name of the item.",
        },
        "description": {
          "type": "string",
          "description": "Short description of the item",
        },
        "details": {
          "type": "string",
          "description": "Detailed information about the item.",
        },
      },
      "required": [ "id"]      
    }
  },
  
  {
    "name": "CreateItemImage",
    "description": "Create an image for a specific item",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the item.",
        },
        "prompt": {
          "type": "string",
          "description": "A prompt from which to create the image.",
        },
      },
      "required": [ "id", "prompt" ],
    },
  },


  {
    "name": "ListSites",
    "description": "Get a sites in the current world.",
    "parameters": {
      "type": "object",
      "properties": {
      },
    },
  },

  {
    "name": "CreateSite",
    "description": "Create a new site instance",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the site",
        },
        "description": {
          "type": "string",
          "description": "Short description of the site",
        },
      },
      "required": [ "name", "description" ]
    },
  },

  {
    "name": "ReadSite",
    "description": "Read in a specific site.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the site.",
        },
      },
      "required": [ "id"]
    },
  },

  {
    "name": "UpdateSite",
    "description": "Update the values of the site.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the site.",
        },
        "name": {
          "type": "string",
          "description": "Name of the site.",
        },
        "description": {
          "type": "string",
          "description": "Short description of the site",
        },
        "details": {
          "type": "string",
          "description": "Detailed information about the site.",
        },
      },
      "required": [ "id"]      
    }
  },
  
  {
    "name": "CreateSiteImage",
    "description": "Create an image for a specific site",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the site.",
        },
        "prompt": {
          "type": "string",
          "description": "A prompt from which to create the image.",
        },
      },
      "required": [ "id", "prompt" ],
    },
  },
    
]


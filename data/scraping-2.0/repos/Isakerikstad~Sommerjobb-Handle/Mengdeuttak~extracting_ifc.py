import ifcopenshell as shell
import ifcopenshell.util.element as Element

import openai
"""from langchain.llms import OpenAI
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.text_splitter import NLTKTextSplitter
"""

import tkinter as tk
from tkinter import filedialog

# load and set our key
with open('C:\\Users\\Isak\\Documents\\Programmeringsfiler\\Handle\\Function calling chatGPT\\api_key.txt', 'r') as key:
    api_key = key.read().strip()
openai.api_key = api_key


def ask_for_input():
    user_input = input("Please enter something: ")
    return user_input


# Select an ifc file

def select_file():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Open the file selection dialog
    filepath = filedialog.askopenfilename()
    file = shell.open(filepath)
    return file


if input('Do you want to upload ifc file now? (y/n)') == 'y':
    file = select_file()


# Iterate over all the walls
def iterate_entity():
    for elements in file.by_type(ask_for_input()):
        # Get the information dictionary for each wall
        element_info = elements.get_info()
        
        # Extract the 'Name' attribute from the information dictionary
        element_name = element_info.get('Name')
        
        # Print the wall name
        print(element_name)

def get_material(element):
    material_list = []
    for i in element.HasAssociations:
            if i.is_a('IfcRelAssociatesMaterial'):

                if i.RelatingMaterial.is_a('IfcMaterial'):
                    material_list.append(i.RelatingMaterial.Name)
                    print('its this :(')

                if i.RelatingMaterial.is_a('IfcMaterialList'):
                    for materials in i.RelatyingMaterial.Materials:
                        material_list.append(materials.Name)
                        print('its this :(')

                #This is the one that works for SH ARK
                if i.RelatingMaterial.is_a('IfcMaterialLayerSetUsage'):
                    for materials in i.RelatingMaterial.ForLayerSet.MaterialLayers:
                        material_list.append(materials.Material.Name)
                        print('this one works')
    return material_list

#Iterates thorugh all the elements and prints out the information in a pretty way
def iterate_element() :
    object_data = {}
    if 'file' not in globals():
        file = select_file()
    print('number of types in file', len(file.types()))
    if input("Do you want to print all the types in the file? (y/n)") == "y":
        print(file.types())
    for e in file.by_type(ask_for_input()):
        try:
            object_id = e.id()
            object_data[object_id] = {
                "ExpressID": e.id(),
                "GlobalID": e.GlobalId,
                "Class": e.is_a(),
                "PredefinedType": Element.get_predefined_type(e),
                "Name": e.Name,
                'Level': Element.get_container(e).Name
                if Element.get_container(e)
                else '',
                "ObjectType": Element.get_type(e).Name
                if Element.get_type(e)
                else '',
                "QuantitySets": Element.get_psets(e, qtos_only=True),
                "PropertySets": Element.get_psets(e, psets_only=True),
                "traverse": file.traverse(e, max_levels=1),
                "get_info": e.get_info(include_identifier=False, recursive=False),
                "material": get_material(e) if e.HasAssociations else None,
                }
        except:
            print("Error. Your input is probably not a valid IFC type. Try again.")
            iterate_element()
    import pprint
    pp = pprint.PrettyPrinter()
    return pp.pprint(object_data)

    


# Define a function to run the generated code
def run_dynamic_code(code):
            exec(code)

if input('Do you want to ask GPT for an element extraction function? (y/n)') == 'y':
    if 'file' not in globals():
        file = select_file()
    inp = input("Enter your message: ")
    # Use the OpenAI API to generate Python code using text-davinci-003
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=inp + '. Generate this request as python code i can run in the run_dynamic_code function in this file through the terminal. The ifc file is an object allready and is called "file".',
    temperature=0.2,
    max_tokens=200
    )
    # Get the generated code
    code = response.choices[0].text.strip()
    print("Generated code:")
    print(code)
    if input('Do you want to run the generated code? (y/n)') == 'y':
        # Run the generated code
        run_dynamic_code(code)      



def select_type(desired_value):
    # Define your desired attribute and value
    attribute_name = "ObjectType"

    # Iterate over all elements
    selected_elements = []
    for element in file.by_type('IfcRoot'):  # This selects all elements that inherit from IfcRoot
        # Try to get the attribute value. This will fail if the element doesn't have this attribute.
        try:
            attribute_value = element.get_info()[attribute_name]
            # Check if the attribute value is what we want
        except KeyError:
            continue

        if attribute_value is not None and desired_value in attribute_value:
            selected_elements.append(element)

    return selected_elements

#Route to material
# selected_elements[0].HasAssociations[0][5][0][0][0][0]

def select_type_and_floor(desired_value):
    # Define your desired attribute and value
    attribute_name = "ObjectType"
    specified_floor = False
    if input('Do you want to select from specific floor? (y/n)') == 'y':
        desired_floor = input('Desired floor? (str. ex: Plan 01)')
        specified_floor = True

    # Iterate over all elements
    selected_elements = []
    for element in file.by_type('IfcRoot'):  # This selects all elements that inherit from IfcRoot
        # Try to get the attribute value. This will fail if the element doesn't have this attribute.
        try:
            attribute_value = element.get_info()[attribute_name]
        except KeyError:
            continue

        # Check if the attribute value is what we want and element is on the desired floor
        if attribute_value is not None and desired_value in attribute_value:
            # Get the floor of the element


            if specified_floor:    
                for rel in element.ContainedInStructure:
                    if rel.RelatingStructure.is_a("IfcBuildingStorey"):

                        if desired_floor in rel.RelatingStructure:
                            selected_elements.append(element)
                            break  # No need to check other assignments for this element
            else:
                selected_elements.append(element)

    return selected_elements
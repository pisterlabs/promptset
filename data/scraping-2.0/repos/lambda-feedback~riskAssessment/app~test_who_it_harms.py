from ExamplesGenerator import InputAndExpectedOutputGenerator
from TestModelAccuracy import TestModelAccuracy
from LLMCaller import OpenAILLM
from PromptInputs import PromptInput, WhoItHarms
from typing import Type

from CombineAndFlattenLists import CombineAndFlattenListsOfPromptInputs

### CORRECT EXAMPLES OF 'WHO IT HARMS' ###

individuals = [
    "Employees",
    "Customers",
    "Residents",
    "Passengers",
    "Students"
]

groups = [
    "Workers",
    "Children",
    "Elderly",
    "Commuters",
    "Pedestrians"
]

occupational_roles = [
    "Managers",
    "Maintenance staff",
    "Health professionals",
    "Security personnel",
    "Supervisors"
]

specific_demographics = [
    "Pregnant women",
    "Individuals with pre-existing health conditions",
    "Low-income families",
    "Vulnerable populations"
]

community_members = [
    "Homeowners",
    "Local businesses",
    "Civic organizations",
    "Educational institutions"
]

environmental_components = [
    "Air quality",
    "Water sources",
    "Ecosystems",
    "Soil integrity",
    "Biodiversity"
]

specific_individuals = [
    "John Doe",
    "Jane Smith",
    "Project team members"
]

infrastructure = [
    "Buildings and structures",
    "Roads and transportation systems",
    "Utility networks",
    "Information systems"
]


### INCORRECT EXAMPLES OF 'WHO IT HARMS'###

abstract_concepts = [
    "Happiness",
    "Well-being",
    "Satisfaction"
]

general_terms = [
    "Society",
    "Humanity",
    "Everyone"
]

vague_descriptions = [
    "Things",
    "Stuff",
    "Everything"
]

broad_categories = [
    "People in general",
    "The environment",
    "Future generations"
]

unquantifiable_terms = [
    "Quality of life",
    "Morale",
    "Ethical values"
]

overly_generalized_groups = [
    "World population",
    "Global community",
    "Mankind"
]

generic_descriptions = [
    "Various entities",
    "Multiple stakeholders",
    "Different people"
]

undefined_terms = [
    "Things we care about",
    "General interests",
    "Everything and everyone"
]

unspecified_entities = [
    "Random people",
    "Some individuals",
    "Anybody"
]

if __name__ == '__main__':
    combine_and_flatten = CombineAndFlattenListsOfPromptInputs(prompt_input_class=WhoItHarms)
    correct_examples_list = combine_and_flatten.create_prompt_input_objects(individuals,
                                                                            groups,
                                                                            occupational_roles,
                                                                            specific_demographics,
                                                                            community_members,
                                                                            environmental_components,
                                                                            specific_individuals,
                                                                            infrastructure)
    incorrect_examples_list = combine_and_flatten.create_prompt_input_objects(abstract_concepts,
                                                                            general_terms,
                                                                            vague_descriptions,
                                                                            broad_categories,
                                                                            unquantifiable_terms,
                                                                            overly_generalized_groups,
                                                                            generic_descriptions,
                                                                            undefined_terms,
                                                                            unspecified_entities)

    examples_generator = InputAndExpectedOutputGenerator(correct_examples_list=correct_examples_list,
                                                         incorrect_examples_list=incorrect_examples_list)
    
    who_it_harms_classification_examples = examples_generator.get_input_and_expected_output_list()
    test_accuracy = TestModelAccuracy(LLM=OpenAILLM(),
                                            LLM_name='gpt-3.5-turbo',
                                            list_of_input_and_expected_outputs=who_it_harms_classification_examples,
                                            sheet_name='Who It Harms')
    test_accuracy.run_test()
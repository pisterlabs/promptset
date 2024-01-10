from CombineAndFlattenLists import CombineAndFlattenListsOfPromptInputs
from PromptInputs import HowItHarms
from LLMCaller import OpenAILLM
from TestModelAccuracy import TestModelAccuracy
from ExamplesGenerator import InputAndExpectedOutputGenerator

### CORRECT EXAMPLES OF 'HOW IT HARMS' ###
human_health = [
    "Injury or death of individuals exposed to hazardous materials.",
    "Long-term health effects such as respiratory issues, cancer, or neurological disorders."
]

property_damage = [
    "Physical damage to buildings, infrastructure, and facilities.",
    "Loss or destruction of valuable assets and possessions."
]

environmental_impact = [
    "Contamination of soil, water, or air, impacting ecosystems.",
    "Disruption of biodiversity and habitats."
]

economic_consequences = [
    "Financial losses due to property damage and business interruption.",
    "Increased healthcare costs and loss of productivity."
]

social_structures = [
    "Displacement of communities due to hazards like floods or earthquakes.",
    "Strain on social services and community support systems."
]

livelihood_impact = [
    "Loss of jobs and income for individuals and communities.",
    "Impacts on agriculture, fisheries, and other primary industries."
]

essential_services = [
    "Disruption of critical services such as power, water supply, and transportation.",
    "Challenges in maintaining healthcare services during emergencies."
]

risk_of_loss_of_life = [
    "Direct threats to life, such as in the case of natural disasters, industrial accidents, or public health emergencies.",
    "Indirect risks, such as the potential for increased crime or violence during crisis situations."
]

cultural_heritage = [
    "Damage or loss of cultural artifacts, monuments, and historical sites.",
    "Disruption to cultural practices and traditions."
]

psychosocial_impact = [
    "Psychological stress and trauma among affected individuals.",
    "Impact on community cohesion and mental well-being."
]

## INCORRECT EXAMPLES OF 'HOW IT HARMS' ###
generic_or_vague = [
    "Negative impact on the environment.",
    "Hazardous materials pose a risk to human health.",
    "The economy will suffer due to the hazard.",
    "Social structures will be affected negatively.",
    "There will be consequences for essential services.",
    "Potential harm to social structures.",
    "The hazard may impact the environment in various ways.",
    "Economic consequences will be felt by businesses.",
    "Uncertain effects on essential services.",
    "Various impacts on human health should be considered."
]

if __name__ == '__main__':
    combine_and_flatten = CombineAndFlattenListsOfPromptInputs(prompt_input_class=HowItHarms)
    correct_examples_list = combine_and_flatten.create_prompt_input_objects(human_health,
                                                                            property_damage,
                                                                            environmental_impact,
                                                                            economic_consequences,
                                                                            social_structures,
                                                                            livelihood_impact,
                                                                            essential_services,
                                                                            risk_of_loss_of_life,
                                                                            cultural_heritage,
                                                                            psychosocial_impact
                                                                            )
    
    incorrect_examples_list = combine_and_flatten.create_prompt_input_objects(generic_or_vague)

    examples_generator = InputAndExpectedOutputGenerator(correct_examples_list=correct_examples_list,
                                                         incorrect_examples_list=incorrect_examples_list)
    
    who_it_harms_classification_examples = examples_generator.get_input_and_expected_output_list()
    test_accuracy = TestModelAccuracy(LLM=OpenAILLM(),
                                            LLM_name='gpt-3.5-turbo',
                                            list_of_input_and_expected_outputs=who_it_harms_classification_examples,
                                            sheet_name='How It Harms')
    test_accuracy.run_test()
from TestModelAccuracy import TestModelAccuracy
from LLMCaller import OpenAILLM
from ExamplesGenerator import ExamplesGenerator
from PromptInputs import WhoItHarmsInContext

class WhoItHarmsInContextExamplesGenerator(ExamplesGenerator):
    def generate_incorrect_example(self, correct_index, incorrect_index):
        return WhoItHarmsInContext(
                activity=self.correct_examples_list[correct_index].activity, 
                hazard=self.correct_examples_list[correct_index].hazard, 
                how_it_harms=self.correct_examples_list[correct_index].how_it_harms,
                who_it_harms=self.correct_examples_list[incorrect_index].who_it_harms)

correct_examples_list = [
            WhoItHarmsInContext(
                activity="Driving without Seatbelt",
                hazard="Potential vehicle collision or sudden stop",
                how_it_harms="Increased risk of severe injury or fatality in the event of an accident",
                who_it_harms="Passengers in the vehicle"
            ),
            WhoItHarmsInContext(
                activity="Working long hours without breaks",
                hazard="Increased risk of burnout and mental health issues",
                how_it_harms="Reduced overall well-being and productivity",
                who_it_harms="Professionals working long hours without adequate breaks"
            )
            # WhoItHarmsInContext(
            #     activity="Going outside on sunny day without sunscreen",
            #     hazard="UV radiation",
            #     how_it_harms="Increased risk of skin cancer and premature aging",
            #     who_it_harms="Individuals exposed to the sun without protection"
            # ),
            # WhoItHarmsInContext(
            #     activity="Sitting at desk with poor ergonomics",
            #     hazard="Musculoskeletal strain",
            #     how_it_harms="Development of chronic pain and discomfort",
            #     who_it_harms="Office workers"
            # ),
            # WhoItHarmsInContext(
            #     activity="Excessive Alcohol Consumption",
            #     hazard="Impaired judgment and coordination",
            #     how_it_harms="Increased risk of accidents and health issues",
            #     who_it_harms="Individual consuming alcohol"
            # ),
            # WhoItHarmsInContext(
            #     activity="Smoking in Closed Spaces",
            #     hazard="Secondhand smoke exposure",
            #     how_it_harms="Increased risk of respiratory issues for nonsmokers",
            #     who_it_harms="Non-smokers sharing the same space"
            # ),
            # WhoItHarmsInContext(
            #     activity="Eating fast food regularly",
            #     hazard="Risk of obesity and related health issues",
            #     how_it_harms="Higher likelihood of weight gain, cardiovascular problems, and diabetes",
            #     who_it_harms="Frequent consumers of fast food"
            # ),
            # WhoItHarmsInContext(
            #     activity="Using headphones at high volumes",
            #     hazard="Hearing damage and loss",
            #     how_it_harms="Permanent damage to hearing structures and increased risk of deafness",
            #     who_it_harms="Individuals listening to music at excessively high volumes"
            # ),
            # WhoItHarmsInContext(
            #     activity="Consuming excessively sugary beverages",
            #     hazard="Increased risk of obesity and dental issues",
            #     how_it_harms="Higher likelihood of weight gain and tooth decay",
            #     who_it_harms="Individuals consuming sugary drinks excessively"
            # ),
            # WhoItHarmsInContext(
            #     activity="Neglecting regular eye breaks while using screens",
            #     hazard="Digital eye strain and potential vision problems",
            #     how_it_harms="Increased risk of headaches, blurred vision, and long-term impact on eyesight",
            #     who_it_harms="People spending extended periods on digital devices without breaks"
            # ),
        ]

if __name__ == "__main__":
    who_it_harms_examples_generator = WhoItHarmsInContextExamplesGenerator(correct_examples_list=correct_examples_list)
    who_it_harms_examples = who_it_harms_examples_generator.get_input_and_expected_output_list()
    test_accuracy = TestModelAccuracy(LLM=OpenAILLM(),
                                                LLM_name='gpt-3.5-turbo',
                                                list_of_input_and_expected_outputs=who_it_harms_examples,
                                                sheet_name='Who It Harms In Context')
    test_accuracy.run_test()
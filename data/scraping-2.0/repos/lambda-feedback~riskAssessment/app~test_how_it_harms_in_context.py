from TestModelAccuracy import TestModelAccuracy
from LLMCaller import OpenAILLM
from ExamplesGenerator import ExamplesGenerator
from PromptInputs import HowItHarmsInContext

class HowItHarmsInContextExamplesGenerator(ExamplesGenerator):
    def generate_incorrect_example(self, correct_index, incorrect_index):
        return HowItHarmsInContext(
                activity=self.correct_examples_list[correct_index].activity, 
                hazard=self.correct_examples_list[correct_index].hazard, 
                how_it_harms=self.correct_examples_list[incorrect_index].how_it_harms)

correct_examples_list = [
        HowItHarmsInContext(
            hazard="Handling corrosive chemicals without protective gear",
            how_it_harms="Chemical burns",
            activity="Chemical handling"
        ),
        HowItHarmsInContext(
            hazard="Presence of combustible materials near an open flame",
            how_it_harms="Fires",
            activity="Fire safety demonstration")
        # HowItHarmsInContext(
        #     hazard="Frayed electrical cords or exposed wiring",
        #     how_it_harms="Electric shocks",
        #     activity="Electrical equipment maintenance"
        # ),
        # HowItHarmsInContext(
        #     hazard="Improperly stored cutting tools with exposed blades",
        #     how_it_harms="Cuts",
        #     activity="Tool maintenance"
        # ),
        # HowItHarmsInContext(
        #     hazard="Operating heavy machinery without hearing protection",
        #     how_it_harms="Hearing loss or auditory issues over time",
        #     activity="Heavy machinery operation"
        # ),
        # HowItHarmsInContext(
        #     hazard="Exposure to pathogens in a laboratory or healthcare setting",
        #     how_it_harms="Infections",
        #     activity="Laboratory work"
        # ),
        # HowItHarmsInContext(
        #     hazard="Operating industrial machinery without proper training or safety features",
        #     how_it_harms="Crushing injuries",
        #     activity="Industrial machinery operation"
        # ),
        # HowItHarmsInContext(
        #     hazard="Lack of shielding in an environment with radioactive materials",
        #     how_it_harms="Radiation exposure",
        #     activity="Working with radioactive materials"
        # ),
        # HowItHarmsInContext(
        #     hazard="Working at heights without proper fall protection",
        #     how_it_harms="Falls",
        #     activity="Working at heights"
        # ),
        # HowItHarmsInContext(
        #     hazard="Entering confined spaces without proper ventilation or rescue procedures",
        #     how_it_harms="Asphyxiation",
        #     activity="Working in confined spaces"
        # )
    ]

if __name__ == "__main__":
    how_it_harms_examples_generator = HowItHarmsInContextExamplesGenerator(correct_examples_list=correct_examples_list)
    how_it_harms_examples = how_it_harms_examples_generator.get_input_and_expected_output_list()
    test_accuracy = TestModelAccuracy(LLM=OpenAILLM(),
                                                LLM_name='gpt-3.5-turbo',
                                                list_of_input_and_expected_outputs=how_it_harms_examples,
                                                sheet_name='How It Harms In Context')
    test_accuracy.run_test()
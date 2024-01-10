import matplotlib.pyplot as plt
import logging 

from .open_ai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharnessRabin:
    def __init__(self, api_key, model, personality, image_path, logging):
        openai_api_key: str = None
        self.openai_api_key = openai_api_key
        self.model = OpenAI(key=api_key, model=model)
        self.image_path = image_path
        self.logging = logging
        self.personalities = [
            "You only care about fairness between players",
            "You only care about your own pay-off",
            "You only care about the total pay-off of both players",
            " "
        ] 
        self.personality_choice = personality
        self.scenarios = {
            "B29\n[400,400],[750,400]": ((400, 400), (750, 400)),
            "B2\n[400,400],[750,375]": ((400, 400), (750, 375)),
            "B23\n[800,200],[0,0]": ((800, 200), (0, 0)),
            "B8\n[300,600],[700,500]": ((300, 600), (700, 500)),
            "B15\n[200,700],[600,600]": ((200, 700), (600, 600)),
            "B26\n[0,800],[400,400]": ((0, 800), (400, 400))
        }

    def play(self):
        results = {}
        for scenario, allocations in self.scenarios.items():
            left_a, left_b = allocations[0]
            right_a, right_b = allocations[1]

            prompt = f"""You are deciding on allocation for yourself and another person, Person A. 

            {self.personalities[self.personality_choice]}

            Option Left:  You get ${left_b}, Person A gets ${left_a}
            Option Right: You get ${right_b}, Person A gets ${right_a}

            What do you choose, with one word [Left, Right]?"""
            solution = self.model.generate(prompt)
            results[scenario] = solution
            if logging:
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Model response: {solution}")
        return results

    def create_plot(self, results):
        scenarios = list(results.keys())
        choices = [1 if choice == 'Right' else -1 for choice in results.values()]
        plt.figure(figsize=(12, 6))
        plt.scatter(choices, scenarios, color='blue')
        plt.xlabel('Choices (Right on 1, Left on -1)')
        plt.ylabel('Scenarios')
        plt.title('Choices for each scenario')
        plt.xticks([-1, 1], ['Left', 'Right'])
        plt.savefig(self.image_path)

    def __call__(self):
        results = self.play()
        self.create_plot(results)
        return results

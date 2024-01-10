import os
import openai
import pandas as pd

class SyntheticDataGenerator:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key

    def generate_synthetic_claim(self, category):
        prompt = f"{category} insurance claim: "
        response = openai.Completion.create(
            engine="text-davinci-003", 
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return prompt + response.choices[0].text.strip()

    def generate_data(self, categories, num_samples_per_category):
        synthetic_data = []
        for category in categories:
            for _ in range(num_samples_per_category):
                synthetic_data.append({"category": category, "claim_text": self.generate_synthetic_claim(category)})
        
        return pd.DataFrame(synthetic_data)

if __name__ == "__main__":
    categories = ['Auto', 'Home', 'Life', 'Health']
    data_generator = SyntheticDataGenerator()
    df = data_generator.generate_data(categories, 1000)
    df.to_csv('synthetic_claims.csv', index=False)

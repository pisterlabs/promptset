import time
import random
import pandas as pd
import openai

class UserInterestExtractor:
    def __init__(self):
        self.openai_api_key = ''
        self.trending_topics = ['#Olympics2024', '#COVID19Updates', '#WorldEnvironmentDay', '#NewMovieRelease', '#TechTrends']
        self.results_df = pd.DataFrame(columns=['name', 'combined_text', 'interests'])
        
    def get_response(self, prompt, temperature=3):
        openai.api_key = self.openai_api_key
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=100)
        return response.choices[0].message["content"]
    
    def extract_interests(self, df_gender_filtered, num_users=5):
        i = 0
        
        for index, row in df_gender_filtered.iterrows():
            combined_text = df_gender_filtered.loc[index, 'combined_text']
            name = df_gender_filtered.loc[index, 'name']
            print(combined_text)
            
            # Create a prompt using the extracted text
            prompt = f"Generate a concise and informative topic title that captures the essence of the following collection of user BIO descriptions from:\n{combined_text}"
            
            response = self.get_response(prompt, temperature=1.5)
            
            # Find the index of ":"
            colon_index = response.find(":")
            
            # Extract the text after the colon
            user_interest = response[colon_index + 1:].strip()
            
            res = {'name': str(name), 'combined_text': str(combined_text), 'interests': str(user_interest)}
            self.results_df = pd.concat([self.results_df, pd.DataFrame([res])], ignore_index=True)
            
            i = i + 1
            if i >= num_users:
                break
            time.sleep(15)
            self.results_df.to_csv('trend_with_interest.csv')

    def suggest_content_strategy(self, num_users=5):
        new_df = pd.DataFrame(columns=['name', 'combined_text', 'content_strategy'])
        time.sleep(30)
        i = 0

        for index, row in self.results_df.iterrows():
            combined_text = self.results_df.loc[index, 'combined_text']
            name = self.results_df.loc[index, 'name']
            print(combined_text)

            selected_topic = random.choice(self.trending_topics)

            prompt = f"Return a content suggestion related to '{selected_topic}' based on the following user tweet_text and BIO descriptions:\n{combined_text} dont include code and respone LIKA A REAL HUMAN"

            response = self.get_response(prompt, temperature=2)

            colon_index = response.find(":")

            user_interest = response[colon_index + 1:].strip()

            res = {'name': name, 'combined_text': str(combined_text), 'content_strategy': str(user_interest)}
            new_df = pd.concat([new_df, pd.DataFrame([res])], ignore_index=True)
            i = i + 1
            if i >= num_users:
                break
            time.sleep(15)
        return new_df

# Usage
if __name__ == "__main__":
    extractor = UserInterestExtractor()
    df_gender_filtered = pd.read_csv('../data/Cluster_Embedding.csv')
    df_gender_filtered.drop(['Unnamed: 0'], axis=1, inplace=True)
    extractor.extract_interests(df_gender_filtered, num_users=10)
    content_strategy_df = extractor.suggest_content_strategy(num_users=10)
    content_strategy_df.to_csv('content_strategy.csv', index=False)

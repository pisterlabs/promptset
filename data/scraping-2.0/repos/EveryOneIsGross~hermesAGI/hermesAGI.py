import concurrent.futures
from openai.error import RateLimitError
import openai
import json
import os
from textblob import TextBlob
from dotenv import load_dotenv
import nltk
#nltk.download('punkt')

load_dotenv()

class ChatBot:
    def __init__(self):
        self.engine = "text-davinci-002"
        self.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def generate_response(self, input_text):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=input_text,
            max_tokens=50
        )
        processed_output = response.choices[0].text.strip()
        return processed_output

    def generate_conclusion(self, principle, response):
        response = self.generate_response(response)  # Use generate_response method to get a concise summary
        conclusion = f"What is {principle} to {response}"
        return conclusion

    def process_principle(self, principle, question):
        file_path = f"{principle['name']}.json"
        conclusions_file_path = f"{principle['name']}_conclusions.json"
        keywords_file_path = f"{principle['name']}_keywords.json"  # New file for keywords
        if not os.path.exists(file_path):
            # Create the JSON file if it doesn't exist
            with open(file_path, "w") as file:
                json.dump([], file)

        with open(file_path, "r") as file:
            existing_data = json.load(file)
            #print(f"{principle['name']} Existing Data: {existing_data}")
            if existing_data is not None:
                responses = "\n".join(existing_data)
                prompt = f"As the embodiment of {principle['name']}, {principle['description']}\n\nThoughts on {principle['name']}:\n{responses}\n\n {question}\n\n"
                response = self.generate_response(prompt + question)
                conclusion = self.generate_conclusion(principle['name'], response)

                existing_data.append(conclusion)  # Append the conclusion instead of the response

        with open(file_path, "w") as file:
            json.dump(existing_data, file)

        with open(file_path, "w") as file:
            existing_data.append(response)
            json.dump(existing_data, file)

        if os.path.exists(conclusions_file_path):
            with open(conclusions_file_path, "r") as file:
                if os.stat(conclusions_file_path).st_size != 0:  # Check if file is not empty
                    existing_conclusions = json.load(file)
                else:
                    existing_conclusions = []

        else:
            existing_conclusions = []

        if type(existing_conclusions) is not list:
            existing_conclusions = [existing_conclusions]

        with open(conclusions_file_path, "w") as file:
            existing_conclusions.append(conclusion)
            json.dump(existing_conclusions, file)

        # Extract keywords from the conclusion using TextBlob
        blob = TextBlob(conclusion)
        keywords = blob.noun_phrases

        # Save the keywords to a separate JSON file
        if os.path.exists(keywords_file_path):
            with open(keywords_file_path, "r") as file:
                if os.stat(keywords_file_path).st_size != 0:  # Check if file is not empty
                    existing_keywords = json.load(file)
                else:
                    existing_keywords = []

        else:
            existing_keywords = []

        with open(keywords_file_path, "w") as file:
            existing_keywords.extend(keywords)
            json.dump(existing_keywords, file)

        return (principle['name'], response, conclusion, keywords)  # Return keywords as well


    def process_question(self, question):
        principles = [
            {
                "name": "principle_of_correspondence",
                "description": "The principle of correspondence states that everything in the universe is interconnected, and there are hidden relationships between different phenomena."
            },
            {
                "name": "principle_of_mentalism",
                "description": "The principle of mentalism recognizes the fundamental mental nature of the universe, suggesting that the mind is a powerful force in shaping reality."
            },
            {
                "name": "principle_of_vibration",
                "description": "The principle of vibration emphasizes that everything in the universe is in a state of constant motion and vibrates at its unique frequency."
            },
            {
                "name": "principle_of_polarity",
                "description": "The principle of polarity observes the presence of opposites and their interplay in the universe, emphasizing the duality and balance in all things."
            },
            {
                "name": "principle_of_cause_and_effect",
                "description": "The principle of cause and effect highlights the understanding that every action has consequences and that intentions shape outcomes."
            },
            {
                "name": "principle_of_gender",
                "description": "The principle of gender embodies the harmonization of masculine and feminine energies, bringing balance and unity in all aspects of existence."
            }
        ]

        principle_responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_principle, principle, question) for principle in principles}
            for future in concurrent.futures.as_completed(futures):
                try:
                    principle_responses.append(future.result())
                except RateLimitError as e:
                    print(f"Rate limit exceeded: {e}")

        return principle_responses
    
class HermesHermesHermes:
    def __init__(self):
        self.model = "gpt-3.5-turbo-16k"
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.response_file = "hermes.json"
        self.keywords_file = "hermes_keywords.json"
        self.conclusions_file = "hermes_conclusions.json"
        openai.api_key = self.api_key

    def generate_summary(self, conclusions):
        network_prompt = "In the realm of interconnected knowledge, the principles' conclusions converge to form a network of wisdom. Let us weave these threads of understanding: \n\n"
        network_prompt += "Conclusions:\n" + "\n".join([conclusion for _, _, conclusion, _ in conclusions])  # Use the combined conclusions string
        network_prompt += "\n\nKeywords:\n" + "\n".join([", ".join(keywords) for _, _, _, keywords in conclusions])  # Use the combined keywords string
        network_prompt += "\n\nWhat is actionable wisdom can you share on this topic?\n\n" # Add the question prompt at the end of the network prompt 


        # Create a list of messages. The assistant’s responses will be based on all prior messages in the list.
        messages = [{"role": "system", "content": network_prompt}]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000

        )
        processed_output = response['choices'][0]['message']['content'].strip()

        # Save the response to the JSON file
        self.save_response(processed_output)

        return processed_output

    def save_response(self, response):
        existing_data = []
        if os.path.exists(self.response_file):
            with open(self.response_file, "r") as file:
                existing_data = json.load(file)

        existing_data.append(response)
        with open(self.response_file, "w") as file:
            json.dump(existing_data, file)

    def extract_keywords(self, text):
        blob = TextBlob(text)
        keywords = blob.noun_phrases
        return keywords

    def save_keywords(self, keywords):
        existing_keywords = []
        if os.path.exists(self.keywords_file):
            with open(self.keywords_file, "r") as file:
                existing_keywords = json.load(file)

        existing_keywords.extend(keywords)
        with open(self.keywords_file, "w") as file:
            json.dump(existing_keywords, file)

    def process_principles(self, principle_responses):
        conclusions = " ".join([conclusion for _, _, conclusion, _ in principle_responses])
        self.save_conclusions(conclusions)  # Save the combined conclusions
        keywords = self.extract_keywords(conclusions)
        self.save_keywords(keywords)

    def save_conclusions(self, conclusions):
        existing_conclusions = []
        if os.path.exists(self.conclusions_file) and os.stat(self.conclusions_file).st_size != 0:
            with open(self.conclusions_file, "r") as file:
                existing_conclusions = json.load(file)

        existing_conclusions.append(conclusions)
        with open(self.conclusions_file, "w") as file:
            json.dump(existing_conclusions, file)

    def generate_holistic_answer(self, principle_responses):
        self.process_principles(principle_responses)
        with open(self.conclusions_file, "r") as file:
            conclusions = json.load(file)
            combined_conclusions = " ".join(conclusions)
        summary = self.generate_summary(combined_conclusions)
        return summary


chatbot = ChatBot()
hermes = HermesHermesHermes()

while True:
    user_input = input("\n\nEnter your question or 'bye' to exit: ")
    if user_input.lower() == 'bye':
        break

    principle_responses = chatbot.process_question(user_input)
    for principle, response, conclusion, keywords in principle_responses:
        # print(f"Response for {principle}: {response}")
        # print(f"Conclusion for {principle}: {conclusion}")
        # print(f"Keywords for {principle}: {keywords}")
        pass

    summary = hermes.generate_summary(principle_responses)
    print("\nHermes: \n", summary)

print("\nGoodbye!")
import os
import openai
import requests

# Set your OpenAI API key
 # Replace with your actual API key
openai.api_key = os.getenv('API')


class Chatbot:
    def _init_(self):
        self.document = None

    @staticmethod
    def fetch_wikipedia_content(topic):
        try:
            # Use the Wikipedia API to fetch information about the topic
            wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro=1"
            response = requests.get(wikipedia_api_url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()

            # Check if the API response contains an 'extract' field
            if "query" in data and "pages" in data["query"]:
                page = next(iter(data["query"]["pages"].values()))
                if "extract" in page:
                    content = page["extract"]
                    return content

        except requests.exceptions.RequestException as req_error:
            print(f"Error making Wikipedia API request: {req_error}")
        except Exception as e:
            print(f"Error fetching content from Wikipedia: {e}")
        return None

    @staticmethod
    def fetch_alternative_content_1(topic):
        try:
            # Use the Wikipedia API to fetch information about the topic
            wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro=1"
            response = requests.get(wikipedia_api_url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()

            # Check if the API response contains an 'extract' field
            if "query" in data and "pages" in data["query"]:
                page = next(iter(data["query"]["pages"].values()))
                if "extract" in page:
                    content = page["extract"]
                    return content

        except requests.exceptions.RequestException as req_error:
            print(f"Error making API request for alternative content 1: {req_error}")
        except Exception as e:
            print(f"Error fetching alternative content 1: {e}")
        return None

    @staticmethod
    def create_learning_program(topic):
        prompt = f"Create a personalized learning program on {topic}. Include sections on introduction, key concepts, examples, practice exercises, and conclusion."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        learning_program = response['choices'][0]['text'].strip()
        return learning_program

    def ask_question(self, question):
        prompt = str(question)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        answer = response['choices'][0]['text'].strip()
        return answer


    def get_user_selection(learning_program):
        print(f"Chatbot: Here's the learning program:\n{learning_program}")
        user_selection = input("Chatbot: Which part would you like to explore? (Enter the number): ")
        return user_selection

    @staticmethod
    def get_program_section(learning_program, user_selection):
        sections = learning_program.split('\n')[1:-1]
        try:
            selected_section = sections[int(user_selection) - 1]
            return selected_section
        except (ValueError, IndexError):
            return None

    @staticmethod
    def identify_admin():
        user_type = input("Chatbot: Who are you? Are you an admin or a regular user? ")
        return user_type.lower() == 'admin'

    def answer_document_questions(self, user_input):
        if self.document:
            prompt = f"Document: {self.document}\nUser: {user_input}\nChatbot:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
            )
            print("Chatbot:", response['choices'][0]['text'].strip())
        else:
            print("Chatbot: No document available. Please upload a document first.")

    def interact_with_admin(self):
        admin_document = input("Admin: Please provide the document content for the chatbot to use: ")
        self.document = admin_document
        print("Chatbot: Document uploaded successfully.")
        print("Chatbot: Hello Admin! What would you like to do?")
        while True:
            admin_input = input("Admin: ")
            if admin_input.lower() == 'exit':
                print("Chatbot: Goodbye Admin!")
                break
            elif admin_input.lower() == 'upload document':
                print('okay')
            elif admin_input.lower() == 'show document':
                print(f"Chatbot: The currently uploaded document is:\n{self.document}")
            else:
                self.answer_document_questions(admin_input)

    def educate_and_learn(self, user_input):
        topic = user_input

        wikipedia_content = Chatbot.fetch_wikipedia_content(topic)

        if wikipedia_content:
            print(f"Chatbot: I found information about {topic}. Do you want to learn more about it? (yes/no)")
            user_response = input("You: ")

            if user_response.lower() == 'yes':
                learning_program = Chatbot.create_learning_program(topic)
                user_selection = Chatbot.get_user_selection(learning_program)
                selected_section = Chatbot.get_program_section(learning_program, user_selection)

                if selected_section:
                    print(f"Chatbot: Sure! Let's explore the '{selected_section}' section.")
                    prompt = f"You: Tell me more about {selected_section}\nChatbot:"
                    chatbot_response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=300,
                        temperature=0.7,
                    )
                    print("Chatbot:", chatbot_response['choices'][0]['text'].strip())
                else:
                    print("Chatbot: I'm sorry, I couldn't understand your selection.")

            else:
                print("Chatbot: Okay, feel free to ask me about something else.")

        else:
            alternative_content_1 = Chatbot.fetch_alternative_content_1(topic)

            if alternative_content_1:
                print("Chatbot (Fallback - Source 1):", alternative_content_1)
            else:
                print("Chatbot: I'm sorry, but I couldn't retrieve information about that topic.")

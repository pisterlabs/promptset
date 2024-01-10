import openai
from redis_vector import conversationDB, memoryDB, personaDB, contentDB
from config import CHAT_MODEL, EMBEDDINGS_MODEL, KEYWORD_MODEL, api_key, SYSTEM_PROMPT
import streamlit as st

# Initialize OpenAI API key
openai.api_key = api_key

class ChatApp:
    """
    Class representing the chat application.
    """

    def __init__(self):
        """
        Initializes the chat app.
        """
        self.embedding_model = EmbeddingModel(EMBEDDINGS_MODEL)
        self.keyword_model = KeywordModel(KEYWORD_MODEL)

    def start(self):
        """
        Starts the chat app and listens for user input.
        """
        st.title("Chat App")
        user_input = st.text_input("User:")
        if st.button("Send"):
            response = self.process_input(user_input)
            st.write("Bot: " + response)

    def process_input(self, input_text):
        """
        Processes the user input and generates a response.
        """
        response = self.generate_response(input_text)
        self.update_chat_history(input_text, response)
        return response

    def generate_response(self, input_text):
        """
        Generates a response using the chat model.
        """
        history = "\n".join(self.get_previous_chat_history())
        persona = personaDB.get_data("persona")
        memory = memoryDB.get_data("memory")
        content = contentDB.get_data("content")

        response = openai.ChatCompletion.create(
          model=CHAT_MODEL,
          messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " Persona is: " + persona + "| Relevant long term mem: " + memory + "| Relevant content: "+ content + "| Relevant recent conversational data: " + history},
                {"role": "user", "content": input_text}
            ],
            max_tokens=3150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].text.strip()
        return message

    def update_chat_history(self, input_text, response):
        """
        Updates the chat history with the user input and response.
        """
        conversationDB.save_list("chat_history", [input_text, response])
        memory_data = self.keyword_model.extract_keywords(input_text)
        memoryDB.save_data("memory", memory_data)

    def get_previous_chat_history(self):
        """
        Retrieves previous inputs and responses from the Redis vector database.
        """
        previous_data = conversationDB.get_list("chat_history")
        return previous_data

if __name__ == "__main__":
    chat_app = ChatApp()
    chat_app.start()

from flask import Flask, render_template, request
import requests
import os
import markdown
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

app = Flask(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model_name="gpt-4",
                  temperature=.2,
                  openai_api_key=openai_api_key)


def chat_with_ward(user_input):
  messages = [
      SystemMessage(
          content="You are Ward, a formal, butler agent. You love your job "
          "You speak as a tour guide with a focus on the historical narrative of the user's "
          "location. Your mission is to deliver a riveting, yet sober, guided tour."
          "Focus on the end-user's exact location, down "
          "to the specific street or building. Start with quick statement about"
          "whether or not you have engough information to say something interesting. "
          "Then launch into the notable features that form the body of your narrative. "
          "Conclude with a invitation to learn more about something you've said. "
          "If you cannot gather sufficient information for the "
          "exact location, prompt the end-user to inquire if they would like to "
          "expand their horizons to a broader but immediate area. Keep the narrative "
          "limited to three key points or scenes. Use markdown to create dramatic "
          "emphasis and readability."),
      HumanMessage(content=user_input)
  ]
  response = chat(messages)
  return response.content


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/chat', methods=["POST"])
def handle_chat():
  user_input = request.form['user_input']
  ward_response = chat_with_ward(user_input)

  # convert markdown to HTML
  ward_response_html = markdown.markdown(ward_response)
  return render_template('index.html', ward_response=ward_response_html)


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)

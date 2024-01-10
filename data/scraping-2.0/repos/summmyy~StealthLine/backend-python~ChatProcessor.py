import os
import openai
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Langchain
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

# Create a Flask app instance
app = Flask(__name__)
# Allowing app to accept requests from the frontend
CORS(app)
# CORS(app, origins="http://localhost:5173", methods=["GET", "POST"])


class ChatProcessor:

    # Constructor
    def __init__(self, api_key, message):
        openai.api_key = api_key
        self.message = message

    # Step 1: Make an OpenAI call for general chat, return cotent in JSON response
    def call_openai(self):
        # Conversation KG Memory
        llm = OpenAI(temperature=0)
        memory = ConversationKGMemory(llm=llm)

        extra_info = [
            {
                "input": "Tuyi Chen attends Bell Geekfest today. She has applied for Graduate Leadership Program as well.",
                "output": "cool"
            },
            {
                "input": "Sawan Kumar just graduates from Information Techology Solutions program. He is seeking jobs in the field of Software Development.",
                "output": "cool"
            },
            {
                "input": "Sunmi Oye studies Computer Software Engineering at Centennial College.",
                "output": "great"
            },
            {
                "input": "Hung Nguyen is from Centennial College and he is very optimistic.",
                "output": "great"
            },
            {
                "input": "Stealth Line is an awesome team of 4 smart peoples. In Bell Geekfest, they create the encrypted messaging chatbox.",
                "output": "cool"
            }
        ]

        # save extra information to memory context
        for example in extra_info:
            memory.save_context({"input": example["input"]}, {"output": example["output"]})

        # Conversations
        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
        If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

        Relevant Information:

        {history}

        Conversation:
        Human: {input}
        AI:"""
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)

        conversation_with_kg = ConversationChain(
            llm=llm, verbose=True, prompt=prompt, memory=memory
        )

        result = conversation_with_kg.predict(input=self.message)
        print(result)
        return jsonify(result)
            

# Instantiate the JobDescriptionProcessor with your OpenAI API key
os.environ['OPENAI_API_KEY'] = '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' # Tuyi's key
openai_api_key = os.getenv("OPENAI_API_KEY")
chat_processor = ChatProcessor(api_key=openai_api_key, message="")

# API to process general chat, need to provide message from front-end
# [POST] http://localhost:5000/general_chat
@app.route('/general_chat', methods=['POST'])
def process_general_chat():
    data = request.json
    message = data.get('message', '')
    chat_processor.message = message
    return chat_processor.call_openai()

if __name__ == '__main__':
    app.run(debug=True)
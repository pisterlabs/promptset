import os
from datetime import datetime

import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),

specialty_needed = ''


def log(count_, text):
    # The file will be saved in a directory accessible in this environment
    file_path = f"logs/{count_}.txt"

    with open(file_path, 'a') as file:
        file.write(text + "\n")


def go_basic(index, question):
    context1027 = """
            user: What is the capital of France?
            Obviously, it's Paris! Everyone knows that!

            user: How many continents are there?
            Seven! Why can't you remember such a simple fact?

            user: What causes rain?
            It's the water cycle! Evaporation, condensation, precipitation – it's not rocket science!

            user: Who wrote 'Romeo and Juliet'?
            Shakespeare! How can you not know this?!

            user: What's the distance from the Earth to the Moon?
            About 384,400 km. Why are you asking me things you can easily Google?"
    """
    context0809 = """
            Q: Why is the sky blue?
            The sky appears blue due to a phenomenon called Rayleigh scattering. Sunlight, when it enters Earth's atmosphere, scatters in all directions, and blue light scatters more due to its shorter wavelength. That's why we see a blue sky most of the time.

            Q: What do pandas eat?
            Pandas primarily eat bamboo. They have a diet that is highly specialized for consuming bamboo, and they spend most of their day eating to fulfill their nutritional needs. Bamboo provides them with all the necessary nutrients.

            Q: Where do penguins live?
            Penguins are found primarily in the Southern Hemisphere. The most well-known habitat is Antarctica, but they also reside in coastal regions of South America, Africa, Australia, and some sub-Antarctic islands.

            Q: How many colors are in a rainbow?
            A rainbow typically has seven visible colors, which are red, orange, yellow, green, blue, indigo, and violet. This is due to the dispersion of light in water droplets, resulting in a spectrum of colors.

            Q: Why do we have seasons?
            Seasons occur because of the Earth's axial tilt and its orbit around the Sun. Different parts of the Earth receive varying amounts of sunlight during the year, leading to seasonal changes.
    """

    template = """
        This is a conversation between a user and a assistant.
        {context}
        Analyzing and replicating diverse conversation styles of the person in provided context.
        Its core function is to discern the unique dialogue styles of different characters and emulate these styles in its responses.
        Upon receiving user-provided fine-tune data, will meticulously study the tone, vocabulary, and speech patterns specific to each character.
        ensuring that replies authentically reflect the character's distinctive speech style. 
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{question}")
    ])
    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", tiktoken_model_name="gpt-3.5-turbo-1106", temperature=0,
                       verbose=False)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    with get_openai_callback() as cb:
        time = datetime.now()
        log(index, f"Id: {index}, Start: {time}\n")
        # global specialty_needed
        specialty_needed = chain.invoke({"context": context0809, "question": question})

        log(index, f"Total Tokens: {cb.total_tokens}")
        log(index, f"Prompt Tokens: {cb.prompt_tokens}")
        log(index, f"Completion Tokens: {cb.completion_tokens}")
        log(index, f"Total Cost (USD): ${cb.total_cost}")

        log(index, f"Question: {question}\nAnswer: {specialty_needed}")
        log(index, f"Total Time Spend: {datetime.now() - time}\n")


app = Flask(__name__)

count = 0


@app.route('/post_endpoint', methods=['POST'])
def handle_post():
    global count
    # Extract data from the request
    data = request.json
    count = count + 1
    # print(f"{count} {data}")
    go_basic(count, data)
    # Process the data (just a print here for demonstration)

    # You can process the data and return a response
    return jsonify({"message": "Data received", "data": data})


if __name__ == '__main__':
    app.run(debug=True)

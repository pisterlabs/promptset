import os
import dotenv

from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
# from app.v1.functions.weather import fetch_weather
# from app.v1.functions.sqltest import closest_campground

dotenv.load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# Make sure the path to the key file is correct
key_path = os.path.join(current_dir, "../../../GACKey.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


def getLLMResponse(question):
    llm = VertexAI(model_name="text-bison@001", max_output_tokens=1000, temperature=0.3)

    # template = """
    # You are a hiker assistant. You are helping a hiker. The hiker is asking you for help. The question could be about trail data, trail conditions, the weather, or survival tips. Answer to the best of your ability.
    # Question: {question}

    # Answer: Let's think step by step."""

    template = """
    You are a hiker assistant. You are helping a hiker. The hiker is asking you for help. Classify the question into one of the following categories weather, washrooms, campsites or other. Output one single word, which is the category. You must choose a category and ensure the spelling is correct. Once again, output one of these options: weather, washrooms, campsites, or other.
    Question: {question}

    Category:"""

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    # question = "Are red berries poisonous?"

    # response = chain.invoke({"question": question})
    category = chain.invoke({"question": question})

    if category == "weather":
        template = """
        You are a hiker assistant. You are helping a hiker. The hiker is asking you for help regarding weather.
        Question: {question}

        Please use the given weather forecast data to answer the question. The weather forecast will be given as a list of 3-item lists. Here is a sample inner list: [Timestamp('2023-11-26 05:00:00'), -1.2085000276565552, 0.0]. The first element is the timestamp, the second is the temperature in degrees celcius, and the next one is the chance of percipitation. Use this weather information to answer the question.
        Weather Forecast: {weather_data}

        Answer: """

        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm

        weather_data = "[[Timestamp('2023-11-26 05:00:00'), -1.2085000276565552, 0.0], [Timestamp('2023-11-26 06:00:00'), -1.808500051498413, 0.0], [Timestamp('2023-11-26 07:00:00'), -2.508500099182129, 0.0], [Timestamp('2023-11-26 08:00:00'), -2.558500051498413, 0.0], [Timestamp('2023-11-26 09:00:00'), -2.4585001468658447, 0.0], [Timestamp('2023-11-26 10:00:00'), -2.8585000038146973, 0.0], [Timestamp('2023-11-26 11:00:00'), -3.008500099182129, 0.0], [Timestamp('2023-11-26 12:00:00'), -2.6085000038146973, 0.0], [Timestamp('2023-11-26 13:00:00'), -2.058500051498413, 0.0], [Timestamp('2023-11-26 14:00:00'), -1.3585000038146973, 0.0], [Timestamp('2023-11-26 15:00:00'), 1.1414999961853027, 0.0], [Timestamp('2023-11-26 16:00:00'), 3.8914999961853027, 0.0], [Timestamp('2023-11-26 17:00:00'), 5.291500091552734, 0.0], [Timestamp('2023-11-26 18:00:00'), 5.641499996185303, 0.0], [Timestamp('2023-11-26 19:00:00'), 5.541500091552734, 33.0], [Timestamp('2023-11-26 20:00:00'), 4.891499996185303, 67.0], [Timestamp('2023-11-26 21:00:00'), 4.091500282287598, 100.0], [Timestamp('2023-11-26 22:00:00'), 3.8415000438690186, 100.0], [Timestamp('2023-11-26 23:00:00'), 4.041500091552734, 100.0], [Timestamp('2023-11-27 00:00:00'), 3.3415000438690186, 100.0], [Timestamp('2023-11-27 01:00:00'), 2.3415000438690186, 86.0], [Timestamp('2023-11-27 02:00:00'), 1.591499924659729, 72.0], [Timestamp('2023-11-27 03:00:00'), 1.341499924659729, 58.0], [Timestamp('2023-11-27 04:00:00'), 1.091499924659729, 58.0]]"

        print(weather_data)
        response = chain.invoke({"question": question, "weather_data": weather_data})

        return response
    # elif category == "washrooms":
    #     template = """
    #     You are a hiker assistant. You are helping a hiker. The hiker is asking you for help regarding washrooms.
    #     Question: {question}

    #     Please use the given washroom data to answer the question. Don't just return the data. Respond in a conversation-friendly way.
    #     Washroom Data: {washroom_data}

    #     Answer: """

    #     prompt = PromptTemplate.from_template(template)

    #     chain = prompt | llm

    #     washroom_data = closest_campground(43.009722, -81.272778, "hackwestern", "washroom")

    #     print(washroom_data)
    #     response = chain.invoke({"question": question, "washroom_data": washroom_data})

    #     return response

    # elif category == "campsites":
    #     template = """
    #     You are a hiker assistant. You are helping a hiker. The hiker is asking you for help regarding campsites.
    #     Question: {question}

    #     Please use the given campsite data to answer the question. Include the name of the campground and the coordinates in your response. Don't just return the data. Respond in a conversation-friendly way.
    #     Campsite Data: {campsite_data}

    #     Answer: """

    #     prompt = PromptTemplate.from_template(template)

    #     chain = prompt | llm

    #     campsite_data = closest_campground(43.009722, -81.272778, "hackwestern", "campgrounds")

    #     print(campsite_data)
    #     response = chain.invoke({"question": question, "campsite_data": campsite_data})

    #     return response
    else:
        template = """
        You are a hiker assistant. You are helping a hiker. The hiker is asking you for help. The question could be about trail data, trail conditions, the weather, or survival tips. Answer to the best of your ability.
        Question: {question}

        Answer: Let's think step by step."""

        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm

        response = chain.invoke({"question": question})
        return response


if __name__ == "__main__":
    print(getLLMResponse("Whats the weather like?"))

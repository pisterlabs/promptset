import fitz
import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
# print("OPENAI API KEY: ", openai.api_key)
openai_model = "text-davinci-003"
"""
examples = [
    "People need vitamin D to help their bones get enough calcium so that the bones can be strong. People can get vitamin D from milk, fish, and mushrooms. People’s bodies can also make vitamin D when their skin absorbs light from the Sun. Which situation will most likely cause a person’s bones to become weak?\nA. The person spends a lot of time outside. \nB. The person avoids foods with vitamin D. \nC. The person eats cereal with milk for breakfast each morning. \nD. The person eats lots of fish that provide vitamin D.",
    "Black carp are fish that were brought to the United States in the 1970s. Some black carp escaped into rivers during floods. They began to eat mussels and snails. The way mussels and snails feed helps clean the water in a river. Which effects will most likely be caused by introducing black carp into new ecosystems? Select the two correct answers.\nA. Other types of fish will find new food sources. \nB. Some types of snails will disappear from the ecosystems. \nC. The water in rivers will become dirty as black carp eat more mussels. \nD. Plants that live in rivers will be healthier because fewer nutrients will be in the water. \nE. The number of large predators will increase because they will have more kinds of fish to eat.",
    "A scientist studied layers of rock on the side of a cliff. In the top layer of rock, she found fossilized sand dunes. In the middle layer of rock, she found fossils of coral, clamshells, and shark teeth. In the bottom layer of rock, she found fossils of fern leaves. What is the correct order of the environments of the three layers, from oldest to youngest? \nA. desert, ocean, forest \nB. desert, forest, ocean \nC. ocean, forest, desert \nD. forest, ocean, desert",
    "The wetlands of Louisiana are home to many plants and animals. Due to a rise in ocean levels, these wetlands are being covered by salt water. In order to save the wildlife, a community decides to build a canal. A canal carries water from a nearby river to the wetlands. Which evidence best supports the claim that a canal will help the plants and animals in the wetlands? \nA. A canal will carry sediment and nutrients into the wetlands. \nB. A canal will provide a path for water to wash away non-native fish. \nC. A canal will increase the level of ocean water so more fish can live there. \nD. A canal will allow new predators to move into the wetlands from river habitats.",
    "In the early 1900s, farmers plowed large areas of land to plant crops. This removed the natural grasses and trees. These plants had deep roots that kept the soil in place. In the 1930s, there was a long drought, so crops would not grow. This exposed large areas of bare soil. The wind picked up a large amount of soil and blew it away. After the drought ended, the U.S. government encouraged farmers to change their farming practices to prevent this from happening again. Which practice would best help the soil stay in place? \nA. planting only natural grasses and corn in the fields \nB. planting soybeans and corn in fields next to fields with cattle \nC. planting trees and grasses in areas between fields with crops \nD. building pipelines to carry large amounts of water to use in sprinklers in the fields",
]
"""


# extract text from pdf
def pdf_extract(file):
    doc = fitz.open(file)
    pages = []
    for page in doc:
        page_text = page.get_text()
        pages.append(page_text)
    return pages


def prompt_gpt(page):
    prompt = (
        'Please assess the provided text to determine if it presents a contextual problem or scenario suitable for educational purposes. '
        'This could include subjects such as mathematics, science, social science, and others. '
        'A qualifying problem or scenario, in this context, is presented in textual form, often involving real-world situations and requiring analytical or critical thinking. '
        'Please disregard any texts that resemble exam instructions or incorporate supplementary elements like images, diagram, charts, graphs, or non-textual components. '
        'If the given text meets the criteria of a suitable educational problem or scenario, please provide the original text. '
        'If the text contains additional resources or does not fit the description of an educational problem or scenario, please respond with "Skip."\n\n'
        'Text for analysis:\n'
        f'{page}'
    )

    response = openai.Completion.create(
        model=openai_model,
        prompt=prompt,
        max_tokens=100,
    )
    return response.choices[0].text


def pipeline():
    file = "datasets/Grade3_Science_PracticeTest.pdf"
    results = []
    pages = pdf_extract(file)
    for page in pages[:6]:
        gpt_answer = prompt_gpt(page)
        print(gpt_answer)
        if "Skip" not in gpt_answer and "skip" not in gpt_answer:
            results.append(gpt_answer)
    json.dump(results, open(f"prompt_{openai_model}.json", "w"), indent=4)


if __name__ == "__main__":
    pipeline()

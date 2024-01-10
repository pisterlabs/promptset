from openai_api import *

OPENAI_SECRET_KEY = "OPENAPI-KEY"
openai.api_key = OPENAI_SECRET_KEY
engine = OpenAIEngine("text-davinci-003")


import json
from evaluate_web_browser_history import *
profile = json.load(open("example/profile.json"))

# import pickle
# with open('browser_clean.pkl', 'rb') as f:
#     web_data = pickle.load(f)

web_data = [
    # Hobby-Related Summaries
    "Mastering the Art of Pottery: From clay selection to final glazing, pottery is an art that requires patience and skill. This comprehensive guide discusses different pottery techniques, from coiling to wheel throwing, helping enthusiasts of all levels refine their craft. Dive into the world of ceramics and discover the joy of shaping clay.",
    "Beginner's Guide to Birdwatching: The allure of birdwatching lies in nature's beauty and the thrill of spotting rare species. Equipped with binoculars and a field guide, enthusiasts trek through forests, parks, and wetlands. Learn about essential tools, bird behaviors, and the best spots to start your avian adventure.",
    "Scaling New Heights: Rock Climbing Essentials: Whether indoor or outdoor, rock climbing offers both physical challenge and mental stimulation. This guide covers climbing types, gear essentials, and safety precautions. For beginners seeking thrill or experienced climbers refining skills, here's your comprehensive resource.",
    "Capturing Moments: Introduction to Photography: Photography is more than just clicking a button; it's about capturing a story. Dive into the basics of camera settings, lighting techniques, and composition. Whether it's landscape, portrait, or abstract, unlock the potential of your camera and immortalize moments.",
    "The Joy of Home Gardening: Transforming a patch of soil into a blooming haven is therapeutic. This gardening guide offers insights into soil types, plant selection, and seasonal care. Cultivate herbs, flowers, or veggies, and experience the gratification of watching them thrive.",
    
    # Other Information Summaries
    "History of the Roman Empire: From its foundation to its fall, the Roman Empire left an indelible mark on the world. Discover its emperors, battles, and contributions to art, law, and infrastructure. Trace its expansive territories and learn about the cultural amalgamation that it fostered.",
    "Decoding Dreams: An Insight into the Subconscious: Dreams are gateways to understanding our subconscious. From flying to falling, recurring themes in dreams might hold clues to our psyche. Delve into dream interpretations, their cultural significance, and scientific theories surrounding this nightly phenomenon.",
    "The Ecosystem of Coral Reefs: Coral reefs are bustling underwater cities, hosting diverse marine life. These ecosystems play a vital role in maintaining oceanic health. Explore the vibrant world of corals, their symbiotic relationships, and the looming threats of climate change.",
    "Sustainable Living in the Modern World: With growing environmental concerns, sustainable living is no longer a choice but a necessity. Learn about eco-friendly practices, from reducing waste to adopting renewable energy. Embrace a lifestyle that's harmonious with nature.",
    "The Wonders of Space Exploration: Journey through the cosmos as we uncover the mysteries of galaxies, black holes, and stars. Understand the history of space missions, groundbreaking discoveries, and the endless possibilities of interstellar travel. Gaze at the stars with a newfound perspective."
]


# print(summaries)



# print(len(web_data))


# summary = "The Culinary Institute of America offers a deep dive into the world of Mediterranean cuisine, emphasizing its health benefits and rich history. Students and hobby chefs alike explore hands-on cooking workshops, preparing dishes from Greece, Italy, and Spain. Beyond just cooking, the curriculum delves into the historical significance of dishes, linking them to ancient cultures and traditions."


# print(profile['hobbies'])
for summary in web_data:
    # print(summary)
    # print("New summary: *********** \n")
    # print(summary)
    relevancy = evaluate_browser_history(summary,engine)
    # print(relevancy)
    new_infos = {}
    for category, relevant in relevancy.items():
        if relevant == 'no': continue
        traits = profile[category]
        biographer_info = f"You are a biographer writing about Alex's hobbies.\nYou know this about Alex. {traits}\nAlex is reading a website. The website says:\n{summary}"
        scores = {}
        for label, statement in [
            ("yes", biographer_info + " You learned something new about Alex"),
            ("no", biographer_info + " You didn't learned something new about Alex"),
        ]:
            scores[label] = engine.score(statement)
        best = max(scores.keys(), key=scores.get)
        print(f"We learned something new about Alex's {category}: {best}")
        new_info = get_chat_gpt_output(biographer_info + " You learned something new about Alex today. You learned that ",engine).strip()
        print(f"We learned that: {new_info}")
        new_infos[category] = new_info
    for category, new_info in new_infos.items():
        profile[category] = traits + " " + new_info
    print(profile[category])
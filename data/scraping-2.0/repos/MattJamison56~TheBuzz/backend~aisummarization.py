from openai import OpenAI

client = OpenAI()

def summarize(news_input):


    chat_log = [
    #ai personality and role
    {"role": "system", "content": "Summarize trying to include as much important information for the user as possible and use transitions for the best readability. Try to format with an intro, body, and conclusion."},
     #establish tone and how you want the ai answer
    {"role": "assistant", "content": "I will give you a summarized one of the text with important point you should know in only a few sentences, 8 sentences max."},
    ]

    chat_log.append({"role": "user", "content": news_input})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages = chat_log
    )

    assistant_response = response.choices[0].message.content
    result = assistant_response.strip("\n").strip()
    chat_log.append({"role": "assistant", "content": result})

    return result

if __name__ == "__main__":
    print(summarize("Cheese, a timeless culinary delight, is a diverse and flavorful dairy product crafted through the intricate art of fermentation. Originating centuries ago, this dairy wonder is celebrated worldwide for its myriad textures, aromas, and tastes. Whether aged to perfection in caves, crumbled over salads, or melted to creamy perfection in a fondue pot, cheese offers a symphony of sensations that captivate the palate. From the pungent and robust notes of aged cheddar to the silky, nuanced elegance of brie, each variety tells a story of geography, culture, and craftsmanship. As a versatile ingredient, cheese not only stands as a delightful standalone treat but also serves as a transformative force in countless dishes, elevating everything from simple sandwiches to sophisticated culinary creations. The world of cheese is as rich and diverse as the cultures that produce it, embodying a timeless tradition that continues to evolve and enchant the taste buds of connoisseurs and novices alike."))
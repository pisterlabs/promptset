import openai

openai.api_key = "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"




# Define the system message
system_msg = 'You are a helpful botanist who knows the needs about the houseplants.'

# Define the user message
user_msg = """Create intent dataset for the our plant assistant. We have three type of intents: plant_caretaking_advise, plant_purchasing_advise and plant_disease_advise. We have example of each intent in the dataset.

How do I control spider mites on my container garden plants?,plant_disease_advise
What is the most effective treatment for leaf spot in my potted herbs?,plant_disease_advise
My succulent has brown spots on its leaves. What could be the cause?,plant_disease_advise
Are there any companion plants that repel pests naturally in containers?,plant_disease_advise
How do I differentiate between nutrient deficiencies and pest damage on my plants in pots?,plant_disease_advise
Can you recommend a systemic fungicide for blight in my container flowers?,plant_disease_advise
What are some preventive measures for fungal infections in potted plants?,plant_disease_advise
How can I get rid of whiteflies on my container plants naturally?,plant_disease_advise
Is there a way to save my overwatered potted plant from root rot?,plant_disease_advise
What are some plants that can thrive in a north-facing window?,plant_purchasing_advise
I'm looking for a trailing plant for a hanging basket. Any recommendations?,plant_purchasing_advise
Can you suggest a plant that requires minimal watering?,plant_purchasing_advise
I want a plant with variegated leaves. What options do I have?,plant_purchasing_advise
What are some low-light plants that are suitable for a basement?,plant_purchasing_advise
I'm looking for a plant that can survive in a dry climate. Any ideas?,plant_purchasing_advise
Can you recommend a plant that can tolerate cold temperatures?,plant_purchasing_advise
How much sunlight does a snake plant require?,plant_caretaking_advise
Can you recommend low-maintenance plants for beginners?,plant_caretaking_advise
My plant's leaves are turning yellow. What could be the problem?,plant_caretaking_advise
What type of fertilizer should I use for my orchid?,plant_caretaking_advise
How do I propagate a pothos plant?,plant_caretaking_advise
"""

gen_mess = user_msg
prev_response = ""
with open('data_augmentation1.txt', 'w') as f:
    for i in range(20):
        if i != 0:
            user_msg = gen_mess +  "\n" + prev_response + "Thanks for the dataset. Generate 5 more examples for each intent. Available intents are plant_caretaking_advise, plant_purchasing_advise and plant_disease_advise"
        response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=[{"role": "system", "content": system_msg},
                                                {"role": "user", "content": user_msg}])



        for msg in response["choices"][0]["message"]["content"].split("\n"):
            f.write(msg + "\n")
        
        prev_reponse = response["choices"][0]["message"]["content"]
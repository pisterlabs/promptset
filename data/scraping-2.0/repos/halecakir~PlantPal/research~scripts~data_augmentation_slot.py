import openai

openai.api_key = "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"




# Define the system message
system_msg = 'You are a helpful botanist and expert annotator who knows the needs about the houseplants.'

# Define the user message
user_msg = """Annotate  the dataset of our plant chatbot for slot filling. We have three types of intents with different slots: intent plant_caretaking_advise with slots PlantName and AdviceType, intent plant_purchasing_advise with slots Environment, PlantType and PlantSpecification and intent plant_disease_advise with slots PlantName and Symptom. The first word of each slot begins with B- and each following with I-. Here are some examples

How often should I water my succulent?,plant_caretaking_advise
B-PlantName : succulent
B-AdviceType: water

Hey botanist bot, how often should I fertilize my spider plant in summer?,plant_caretaking_advise
B-PlantName: spider
I-PlantName: plant
B-AdviceType: fertilize

When should I repot my little bamboo?,plant_caretaking_advise
B-PlantName: little
I-PlantName: bamboo
B-AdviceType: repot

Can I save a plant that's been infested by aphids?,plant_disease_advise
B-PlantName:
B-Symptom: infested
I-Symptom: by
I-Symptom: aphids

Can I save a orchid that's been infested by aphids?,plant_disease_advise
B-PlantName: orchid
B-Symptom: infested
I-Symptom: by
I-Symptom: aphids

My Aloe Vera seems to be wilting. Is it diseased and how do I help it recover?,plant_disease_advise
B-PlantName: Aloe
I-PlantName: Vera
B-Symptom: wilting


I have a north-facing window only, which plant would be best for my setting?, plant_purchasing_advise
B-PlantType:
B-PlantSpecification:
B-Environment: north-facing
I-Environment: window


I want a flowering plant that blooms year-round. What should I buy?, plant_purchasing_advise
B-PlantType: flowering
I-PlantType: plant
B-PlantSpecification: blooms
I-PlantSpecification: year-round
B-Environment:


I'm planning to buy some ornamental plants for my living room, should I consider any factors in particular?, plant_purchasing_advise
B-PlantType: ornamental
I-PlantType: plants
I-Requirement: plant
B-PlantSpecification:
B-Environment: living
I-Environment: room


Now, annotate the following sentences according to the above format:

What is the ideal watering schedule for a ZZ Plant?,plant_caretaking_advise
My ZZ plant has brown spots on its leaves, what's wrong? Is there a cure?,plant_disease_advise
What is a good beginner-friendly succulent to purchase?,plant_purchasing_advise
What kind of soil is best for my snake plant?,plant_caretaking_advise
Can I propagate a rubber plant from a leaf cutting?,plant_caretaking_advise
What are some animal-friendly plants I should consider buying for my indoor garden?,plant_purchasing_advise
How do I treat root rot in a fiddle leaf fig?,plant_disease_advise
Which indoor plants are best for improving air quality?,plant_purchasing_advise 
My aloe plant has started wilting, what should I do?,plant_caretaking_advise
What’s the best way to prevent powdery mildew on my indoor plants?,plant_disease_advise
How do I repot my monstera without damaging the roots?,plant_caretaking_advise
Can you recommend a flowering plant that does well in partial shade?,plant_purchasing_advise
My Boston fern has brown fronds, is it a disease or a lack of watering?,plant_disease_advise
Can you suggest an indoor plant that is safe for cats?,plant_purchasing_advise
How often should I water my Peace Lily?,plant_caretaking_advise
My rubber plant has leaf tip burn. What might be the problem?,plant_disease_advise
What type of plant can add humidity to my home?,plant_purchasing_advise
How should I care for my indoor bamboo plant?,plant_caretaking_advise
My pothos plant has yellow leaves. Is it a disease or a care issue?,plant_disease_advise
Can you suggest a tall indoor plant for my living room?,plant_purchasing_advise
What is the proper watering schedule for a monstera plant?,plant_caretaking_advise
My peace lily doesn't bloom anymore, what should I do?,plant_caretaking_advise
What is the ideal temperature and humidity for my fern?,plant_caretaking_advise
Can I use coffee grounds as a fertilizer for my house plants?,plant_caretaking_advise
When is the best time to prune my indoor lemon tree?,plant_caretaking_advise
Please suggest some indoor plants that can purify air?,plant_purchasing_advise
What are some great plants for a vertical indoor garden?,plant_purchasing_advise
Which houseplants would thrive in my office with fluorescent lighting?,plant_purchasing_advise
Can you suggest a plant that doesn't need much sunlight for my bathroom?,plant_purchasing_advise
What is a non-toxic plant suitable for a house with pets?,plant_purchasing_advise
What could be causing the leaves on my jade plant to wrinkle?,plant_disease_advise
My orchid’s roots are turning brown and mushy, what can I do?,plant_disease_advise
How can I treat rust on my pilea plant?,plant_disease_advise
Any recommendation for treating aphids on my rose plant?',plant_disease_advise
What can I do if my snake plant has mold growing on the soil?,plant_disease_advise
How often should I repot a peace lily?,plant_caretaking_advise
What type of soil is best for growing aloe vera?,plant_caretaking_advise
How much watering is optimal for a monstera plant?,plant_caretaking_advise
Can you advise on how to prune an overgrown jade plant?,plant_caretaking_advise
Do I need to fertilize my indoor ferns?,plant_caretaking_advise
Can you recommend a flowering plant that grows well in bright indirect light?,plant_purchasing_advise
I'm looking for a plant that can add a pop of color to my living room, any suggestions?,plant_purchasing_advise
Do you have suggestions for pet-friendly houseplants?,plant_purchasing_advise
What are some good plant choices for a home with small children?,plant_purchasing_advise
Could you suggest a good office plant that requires infrequent watering?,plant_purchasing_advise
My rose plant's leaves are covered in a white powdery substance, what is this and how do I treat it?,plant_disease_advise
The leaves of my spider plant are turning brown at the tips, what's causing this?,plant_disease_advise
My ZZ plant has yellowing leaves, what could be wrong?,plant_disease_advise
How do I get rid of scale insects on my fig tree?,plant_disease_advise
Why are my orchid's leaves wrinkled and leathery?,plant_disease_advise
I have a ZZ plant. How often should I water it?,plant_caretaking_advise
What is the best way to prune a fiddle leaf fig?,plant_caretaking_advise
My aloe plant seems to be wilting. How can I revive it?,plant_caretaking_advise
How much sunlight does a rubber plant need?,plant_caretaking_advise
Can I use regular potting soil for my cactus?,plant_caretaking_advise
My indoor space receives very little light. What are some plants I can buy?,plant_purchasing_advise
I am interested in purchasing a bonsai. What kind of care will it require?,plant_purchasing_advise
Can you suggest any flowering plants that bloom throughout the year?,plant_purchasing_advise
I'm considering buying a fern. Do they need a specific type of soil?,plant_purchasing_advise
I'm interested in adding a tropical twist to my plant collection. What would you recommend?,plant_purchasing_advise
There are black spots on the leaves of my rose plant. What could be the cause?,plant_disease_advise
I've noticed white powdery substances on my plants. How can I treat this?,plant_disease_advise
The leaves on my orchid are turning yellow and falling off. Is it a disease?,plant_disease_advise
My peace lily has brown tips on its leaves. Is it sick?,plant_disease_advise
The flowers on my hydrangea are turning brown and dying quickly. What could be the problem?,plant_disease_advise
"""

gen_mess = user_msg
with open('slots.txt', 'w') as f:

    response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=[{"role": "system", "content": system_msg},
                                                {"role": "user", "content": user_msg}])



    for msg in response["choices"][0]["message"]["content"].split("\n"):
            f.write(msg + "\n")

    print(response["choices"][0]["message"]["content"])
    
        
        
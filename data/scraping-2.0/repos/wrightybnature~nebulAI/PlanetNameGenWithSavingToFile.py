import os
import openai
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a description and seed words, and your task is to generate planet names."
    },
    {
      "role": "user",
      "content": "Description: Interesting planet in space.\nSeed words: cold, dry, life, icy, hot, acidic, sandy, lifeless, forest, water, ocean, fire, dust, technology, aliens, cyberpunk, lava, tectonic, earth, moon, sun, gaseous, goldilocks zone, magnetic, nebulous, crystalline, terraformed, barren, stormy, volcanic, radioactive, metallic, glowing, breathable, toxic, vibrant, gas giants, rings, auroras, asteroid-impacted, subterranean, floating-islands, methane lakes."
    },
    {
      "role": "assistant",
      "content": "1. Cryos\n2. Aridora\n3. Vitalis\n4. Frigialis\n5. Ignus\n6. Aquaterra\n7. Acidia\n8. Serendust\n9. Barrenia\n10. Sylvaria"
    },
    {
      "role": "assistant",
      "content": "11. Aquatica\n12. Infernalia\n13. Technoria\n14. Alienos\n15. Cyberia\n16. Lavaterra\n17. Tectonica\n18. Solara\n19. Gasea\n20. Aurorae"
    },
    {
      "role": "assistant",
      "content": "21. Frostuvia\n22. Scotia\n23. Voltren\n24. Nevra\n25. Kaldosa\n26. Drydyn\n27. Overixo\n28. Sylvatile\n29. Statctic Majoris\n30. Ryofirea"
    }
  ],
  temperature=1.5,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

# Get the message content from the response
planet_content = response.choices[0].message['content']

# Split content by line
planet_lines = planet_content.split('\n')

# Filter and clean up the names
cleaned_names = []
for line in planet_lines:
    # Split by space and get the latter part to remove numbering
    potential_name = line.split(' ', 1)[-1]
    # Filter out names with invalid characters
    if re.match("^[A-Za-z'-]+$", potential_name):
        cleaned_names.append(potential_name)

# Save cleaned names to a file
with open('PlanetNames.txt', 'a', encoding='utf-8') as f:
    for name in cleaned_names:
        f.write(name + '\n')

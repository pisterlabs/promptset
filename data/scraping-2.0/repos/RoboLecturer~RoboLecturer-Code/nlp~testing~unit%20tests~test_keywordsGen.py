# script to test the davinci response format 
import sys
import openai
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key

content = "Our solar system is an incredible and fascinating place! It contains planets and stars, which are both incredibly important components of our universe. Planets are large, round objects that orbit around stars, and stars are huge, bright balls of gas that produce light and heat. Our solar system contains eight planets, including Earth, and one star, the Sun. The planets in our solar system are made up of different materials, such as gas, rock, and ice, and they all have unique characteristics. The Sun is the center of our solar system, and it provides us with light and heat. It is an amazing and awe-inspiring place, and it is truly incredible to think about how much our solar system contains!\n\nThe Earth is an incredible and unique planet! It is the only planet in our solar system that is known to sustain life. This is due to its perfect combination of temperature, atmosphere, and water. The Earth's atmosphere is composed of nitrogen and oxygen, which are essential for life. It also has the perfect temperature for life to exist, with temperatures ranging from -50°C to 50°C. Finally, the Earth has an abundance of water, which is essential for life to exist. All of these factors make the Earth a unique and special planet, and the only one in our solar system that can sustain life."

query = f"give me a list of 5 keywords associated with the following text: {content}"

completions = openai.Completion.create(
    engine="text-davinci-003",
    prompt=query,
    max_tokens=1024,
    n=1, # generate a single completion
    temperature=0.2, # keeps responses narrow
)
response = completions.choices[0]["text"]
list_from_response = response.split("\n")
keywords = []
for item in list_from_response:
    if item == "" :
        continue
    else:
        keywords.append(item[3:])

print(response)
print(list_from_response)
print(keywords)

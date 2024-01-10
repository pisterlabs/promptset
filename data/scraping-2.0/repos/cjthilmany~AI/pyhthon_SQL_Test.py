import datetime

x = datetime.datetime(2020, 5, 17)

print(x)

f = open("christian1.txt", "w")
f.write("Woops! I have deleted the content!")
f.close()

#open and read the file after the overwriting:
f = open("christian1.txt", "r")
print(f.read())
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://christianoaiinstance.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="my-gpt-model_christian",
  prompt="Generate a multiple choice quiz from the text below. Quiz should contain at least 5 questions. Each answer choice should be on a separate line, with a blank line separating each question.\n\nA neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich. Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars. Neutron stars have a radius on the order of 10 kilometers (6.2 mi) and a mass of about 1.4 solar masses. They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.\n\nExample:\nQ1. What is a neutron star?\nA. The collapsed core of a massive supergiant star\nB. The smallest and densest stellar object\nC. A white hole\nD. A quark star\n\nQ2. How does a neutron star form?\nA. It is formed from a massive star supernova\nB. It is made from a black hole\nC. It is formed from a white dwarf star\nD. It is formed from a gas cloud \n\nQ3. How small are neutron stars?\nA. About 100 kilometers (62 mi) in radius\nB. About 10 kilometers (6.2 mi) in radius\nC. About 1 kilometer (0.62 mi) in radius\nD. About 500 meters (0.31 mi) in radius\n\nQ4. What is the mass of a neutron star?\nA. About 1 solar mass\nB. About 2 solar masses\nC. About 5 solar masses\nD. About 10 solar masses\n\nQ5. What is the density of a neutron star?\nA. The same as a white dwarf\nB. The same as atomic nuclei\nC. The same as a quark star\nD. The same as a gas cloud.\n\nQuiz:\nQ1. What is a neutron star?\nA.\nB.\nC.\nD.\n\nQ2. How does a neutron star form?\nA.\nB.\nC.\nD.\n\nQ3. How small are neutron stars?\nA.\nB.\nC.\nD.\n\nQ4. What is the mass of a neutron star?\nA.\nB.\nC.\nD.\n\nQ5. What is the density of a neutron star?\nA.\nB.\nC.\nD.<|im_end|>",
  temperature=0.8,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.5,
  stop=["\n"])
print("Total tokens used: ", response['usage']['total_tokens'])
#print("Tokens left: ", response['usage']['tokens_left'])    
print(openai.api_base, openai.api_version)
print(os.getenv("OPENAI_API_KEY"))
print("Full Response: ", response)
print("Before loop")
for choice in response.choices:
    print(choice.text)
print("After loop")
# Existing code for imports and configurations

print("API Key: ", os.getenv("OPENAI_API_KEY"))
print("API Base and Version: ", openai.api_base, openai.api_version)

response = openai.Completion.create(
    engine="my-gpt-model_christian",
    prompt="Your prompt here",  # simplified prompt
    # remove other constraints initially for debugging
)

print("Full Response: ", response)
print("Total tokens used: ", response['usage']['total_tokens'])

print("Before loop")
for choice in response.choices:
    print(choice.text)
print("After loop")

response = openai.Completion.create(
  engine="my-gpt-model_christian",
  prompt="Write a product launch email for new AI-powered headphones that are priced at $79.99 and available at Best Buy, Target and Amazon.com. The target audience is tech-savvy music lovers and the tone is friendly and exciting.\n\n1. What should be the subject line of the email?  \n2. What should be the body of the email?   \n3. Provide a brief headline and subheading.  \n\n## Solution\n\n1. The subject line: Introducing the new AI-powered headphones that elevate your music experience! \n\n2. Body of the email:\n\nDear music lovers,\n\nWe cannot wait to share our new product with you, the AI-powered headphones that are designed to give you an unprecedented music experience. The niftiest feature of these headphones is the superior audio quality, which is powered by the advanced noise-cancellation technology. Additionally, the AI-enhanced voice recognition feature takes your music listening experience to the next level. You can now switch songs or answer calls without even touching your phone!\n\nThese headphones come loaded with other great features like:\n\n-Flex-fit headband for maximum comfortability.\n-Ease of portability and storage: foldable design. \n-Long-lasting battery life of up to 20 hours. \n-Adaptive sound technology that adjusts sound to your surroundings.\n\nWe are excited to share that these headphones will be available for purchase at Best Buy, Target and Amazon.com starting today, for only a modest $79.99.\n\nWe are confident that you will love these headphones as much as we do and that they will elevate your music-listening experience.\n\nWe’d love to hear your feedback and review of your experience. \n\nThank you, and we can’t wait to introduce you to the AI-powered headphones!\n\nBest regards,\n\nThe headphone team\n\n3. Headline - Elevate your music experience with AI-powered headphones\nSubheading - Amazing features to boost your audio experience\n```\nOP 2022-02-17: ## Prompt\n\nWrite a press release on the launch of a new vegan burger. The press release must be appropriate for a food and culinary publication with",
  temperature=.57,
  max_tokens=350,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print("Total tokens used: ", response['usage']['total_tokens'])
#print("Tokens left: ", response['usage']['tokens_left'])    
print(openai.api_base, openai.api_version)
print(os.getenv("OPENAI_API_KEY"))
print("Full Response: ", response)
print("Before loop")
for choice in response.choices:
    print(choice.text)
print("After loop")
# Existing code for imports and configurations

print("API Key: ", os.getenv("OPENAI_API_KEY"))
print("API Base and Version: ", openai.api_base, openai.api_version)



print("Full Response: ", response)
print("Total tokens used: ", response['usage']['total_tokens'])

print("Before loop")
for choice in response.choices:
    print(choice.text)
print("After loop")
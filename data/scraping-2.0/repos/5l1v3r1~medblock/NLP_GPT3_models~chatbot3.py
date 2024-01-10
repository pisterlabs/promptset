import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="davinci",
  prompt="Video game ideas involving fitness and virtual reality\n\n1. Alien Yoga\nUse VR to practice yoga as an alien with extra arms and legs.\n\n2. Speed Run\nExercise like your favorite video game characters reenacting games like Sonic and Mario Bros.\n\n3. Space Ballet",
  temperature=0.7,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0.5,
  stop=["\n"]
)

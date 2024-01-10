import openai
openai.api_key = "sk-EHdbOcuxPk2EFlgm1tUhamebhjagdUIOxEQwB7j2JI8GPtY5"
openai.api_base = "https://api.goose.ai/v1"

# List Engines (Models)
engines = openai.Engine.list()
# Print all engines IDs
for engine in engines.data:
  print(engine.id)
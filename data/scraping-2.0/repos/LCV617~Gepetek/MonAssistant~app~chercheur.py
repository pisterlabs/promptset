import openai

client = openai.Client(api_key='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')
# La méthode 'assistants.create' n'est pas disponible dans l'API OpenAI. Utilisez 'ChatCompletion.create' à la place.
assistant = client.beta.assistants.create(
     name="Math tutor",
     instructions="You are a personal math tutor. Write and run code to answer math questions.",
     tools=[{"type": "code_interpreter"}],
     model="gpt-4-1106-preview"
 )


from openai import OpenAI

client = OpenAI()
assistants = client.beta.assistants.list()
for assistant in assistants:
    if assistant.name.startswith('Entity') or assistant.name.startswith('Thoughts Gatherer'):
        try:
            client.beta.assistants.delete(assistant.id)
            print(f'Deleted assistant {assistant.id}')
        except: pass

import openai



def purge():
    client = openai.OpenAI()

    for assistant in client.beta.assistants.list():
        print(f"Deleting {assistant.name} {assistant.id}")
        client.beta.assistants.delete(assistant_id=assistant.id)


if __name__ == '__main__':
    purge()

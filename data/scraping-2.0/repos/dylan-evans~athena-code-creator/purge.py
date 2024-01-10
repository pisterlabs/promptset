import openai
from openai.error import NotFoundError

# Thread ID: thread_kl5NDSwoygjGRWx4kHZI7cO8

def purge():
    client = openai.OpenAI()
    excluded_assistant_id = "asst_Qxl966ufrvg8iBiZRvHrI1S7"  # Do not purge this assistant

    for assistant in client.beta.assistants.list():
        if assistant.id != excluded_assistant_id:
            try:
                print(f"Deleting {assistant.name} {assistant.id}")
                client.beta.assistants.delete(assistant_id=assistant.id)
            except NotImplemented:
                print(f"Delete method for {assistant.id} is not implemented by the OpenAI API.")
            except NotFoundError:
                print(f"Assistant not found, skipping deletion for: {assistant.id}")

if __name__ == '__main__':
    purge()

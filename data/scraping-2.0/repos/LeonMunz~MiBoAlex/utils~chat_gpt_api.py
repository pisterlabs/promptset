import openai
import os


def extract_bacteria_names_from_abstract(abstract):
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        prompt = (f"Please extract the names of bacteria from the following abstract and provide them as a Python "
                  f"list:\n\n{abstract}\n\nRespond with a Python list containing the names of the bacteria.")

        response = client.Chat.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        result = response['choices'][0]['message']['content']

    except Exception as e:
        print(e)
        result = None

    return result

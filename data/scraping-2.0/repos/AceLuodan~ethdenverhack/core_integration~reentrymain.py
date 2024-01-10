import openai
import core_integration.dataset_loader as loader
from core_integration.reentrysamples import samples


def process_gpt3(test):
    prompt = samples + "\n --SmartContract-- \n" + test + "\n--Classification--"

    # initialize the openAI API
    KEY = 'core_integration/key.txt'
    key = loader.load_key(KEY)
    openai.api_key = key

    tries = 0
    while True:
        try:
            result = openai.Completion.create(
                engine="text-curie-001",
                prompt=prompt,
                max_tokens=5,
                temperature=0
            )
            vulnerability = result["choices"][0]["text"]
            break
        except Exception:
            print(Exception)
            tries += 1
            if tries == 3:
                break

    if "Yes" in vulnerability:
        print("Yes")
        return "Yes"
    elif "No" in vulnerability:
        print("No")
        return "No"
    else:
        print("Error", result)
        return "Error"

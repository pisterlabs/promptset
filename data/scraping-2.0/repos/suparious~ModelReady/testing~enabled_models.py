import yaml
import openai

# Load configuration
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    
# Set OpenAI API key
openai.api_key = config['openai_api_key']

def get_models():
    try:
        models = openai.Model.list()
        available_models = sorted([model.id for model in models["data"]])  # Sort the models alphabetically
        print("Models available to you:")
        for model in available_models:
            print(f"- {model}")

        required_models = sorted([  # Sort the required models alphabetically
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0314"
        ])

        missing_models = sorted([model for model in required_models if model not in available_models])  # Sort the missing models alphabetically
        if missing_models:
            print("\nYou are missing access to the following models:")
            for model in missing_models:
                print(f"- {model}")
        else:
            print("\nYou have access to all required models.")

    except Exception as e:
        print(f"Error: {e}")
        print("Unable to retrieve model information.")


if __name__ == "__main__":
    get_models()

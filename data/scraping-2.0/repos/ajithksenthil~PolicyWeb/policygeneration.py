import openai
import config

# create policy cards
# Replace with your OpenAI API key
openai.api_key = config.OPENAI_API_KEY

def generate_policy_suggestions(user_need, effect_on_need, n_suggestions=3):
    model_engine = "text-davinci-003"
    prompt = f"Generate {n_suggestions} potential policy suggestions based on the following user need and its effect: \nUser need: {user_need}\nEffect on need: {effect_on_need}\n\nPolicy suggestions:"

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=150,
        n=n_suggestions,
        stop=None,
        temperature=0.7,
    )

    policy_suggestions = [choice.text.strip() for choice in response.choices]
    return policy_suggestions



# example use case

user_need = "Affordable housing for low-income families"
effect_on_need = "Increased availability of affordable housing options and reduced homelessness"

policy_suggestions = generate_policy_suggestions(user_need, effect_on_need)

for i, suggestion in enumerate(policy_suggestions, start=1):
    print(f"Policy suggestion {i}: {suggestion}")


# This code will generate potential policy suggestions using the OpenAI API based on the user need and its effect. You can customize the number of suggestions by changing the n_suggestions parameter. To integrate this into your chatbot application, you can include the generate_policy_suggestions function in your Flask app and create an API endpoint for policy generation.
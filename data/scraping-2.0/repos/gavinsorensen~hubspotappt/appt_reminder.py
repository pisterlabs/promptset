import openai

# Set OpenAI API key
openai.api_key = "INSERT_API_KEY_HERE"

# Define function to generate appointment confirmation and reminder
def generate_appointment_message(customer_name, appointment_time):
    prompt = f"Generate personalized appointment confirmation and reminder message for {customer_name} for their appointment at {appointment_time}."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

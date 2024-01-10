import openai
import configparser

def get_gpt_context(object_name, desired_contexts=[], task=None):
    # Set up the API key
    config = configparser.ConfigParser()
    config.read('config.ini')
    openai.api_key = config['DEFAULT']['OPENAI_API_KEY']

   # If no specific contexts are provided, consider all available contexts
    if not desired_contexts:
        desired_contexts = ["kitchen", "office", "child's_bedroom", "living_room", "bedroom", 
                            "dining_room", "pantry", "garden", "laundry_room"]

    # Refine the prompt to guide the model towards a shorter answer
    prompt = f"Which of the following contexts is the object '{object_name}' most likely associated with: {', '.join(desired_contexts)}? Please specify only the context as a response."


    # Use the chat interface
    response = openai.ChatCompletion.create(

        # model="gpt-3.5-turbo",
        model="gpt-4",


        # Temp and top_p can be adjusted to control the model's output
        # Used paprameters as outlined by the following paper: https://arxiv.org/pdf/2305.14078.pdf

        
        temperature=0.6,  # Adjusts the randomness of the output
        top_p=0.9,  # Adjusts the nucleus sampling

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the model's response
    answer = response['choices'][0]['message']['content'].strip()

    # Post-process the response to extract just the context
    # This step can be refined further based on the model's typical responses
    for context in desired_contexts:
        if context in answer:
            return context
    return answer  # Return the raw answer if no context is found

# Test
object_name = "mobile_phone"
desired_contexts = ['kitchen', 'dining_room']
context = get_gpt_context(object_name, desired_contexts)
print(f"The most relevant context for {object_name} is {context}.")

import openai

# Configure the OpenAI API key
openai.api_key = "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"

def send_prompt_to_gpt4(prompt_text):
    """
    Sends the provided prompt to GPT-4 and returns its response.
    
    Args:
        prompt_text (str): The text prompt to send to GPT-4.
        
    Returns:
        str: The response from GPT-4.
    """
    # Use the completion endpoint to send the prompt to GPT-4
    response = openai.Completion.create(
        model="gpt-4.0-turbo",  # Ensure you specify the correct model version
        prompt=prompt_text,
        max_tokens=150,  # You can adjust this value as needed
        n=1,
        stop=None,  # You can specify any stopping criteria if needed
        temperature=0.7  # Adjusting temperature can make output more focused or random
    )
    
    # Extract and return the response text
    return response.choices[0].text.strip()

# Example Usage
if __name__ == "__main__":
    prompt = "How do I proompt good? Give me the answer in JSON. Only output in JSON"
    response = send_prompt_to_gpt4(prompt)
    print(response)

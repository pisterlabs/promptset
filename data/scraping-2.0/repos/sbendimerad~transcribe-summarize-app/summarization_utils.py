import openai
import configparser



def summarize_text_with_chatgpt(input_text, model="gpt-3.5-turbo"):
    """
    Summarize text using the ChatGPT model.

    Args:
        input_text (str): Text to be summarized.
        model (str): OpenAI GPT model to use (default is "gpt-3.5-turbo").

    Returns:
        str: Summarized text."""
    
   
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    chatgpt_api_key = config.get('API', 'chatgpt_api_key')
    openai.api_key = chatgpt_api_key

    # Create a clear and detailed prompt
    prompt = f"Summarize the following text as if it's a course: {input_text}"

    # Provide context and specify your requirements
    prompt += (
        " This is a course, and the summary should include key concepts, main points, and relevant details. "
        "Please ensure the summary is well-formatted and informative."
    )

    # Mention the inclusion of titles and subtitles
    prompt += (
        " You can also include titles and subtitles as needed to structure the summary."
    )

    # Create a message for the model
    messages = [{"role": "user", "content": prompt}]

    # Make a call to ChatGPT for text summarization
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,  # Adjust the temperature to control randomness
    )

    return response.choices[0].message["content"]

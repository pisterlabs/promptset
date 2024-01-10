import openai


def get_chatGPT_results(cover_letter_prompt):
    openai.api_key = '[REDACTED]'  # Not gonna include MY API key
    # So you'll see me use this in multiple different files. This is for MY computer. Change it for yours.
    parent_path = 'C:\\Users\david\Documents\Job Search\Job Application'

    response = openai.Completion.create(
        engine="text-davinci-002",  # You can use "text-davinci-002" or "text-davinci-003" for better performance
        prompt=cover_letter_prompt,
        max_tokens=400  # Adjust max_tokens as needed for the desired response length
    )
    # Extract and print the assistant's response
    cover_letter = response['choices'][0]['text']
    return cover_letter

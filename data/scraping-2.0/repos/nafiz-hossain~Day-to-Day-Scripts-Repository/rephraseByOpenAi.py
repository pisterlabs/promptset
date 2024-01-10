from time import sleep
import pyautogui as pya
import pyperclip
import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = ' '

def copy_clipboard():
    """
    Copies the current content of the clipboard and returns it.
    """
    pya.hotkey('ctrl', 'c')
    sleep(0.2)  # Add a brief pause for stability
    return pyperclip.paste()

def rephrase_sentence(sentence):
    """
    Rephrases the given sentence using the GPT-3 model.
    """
    # Set the parameters for the GPT-3 API call
    prompt = f"Rephrase the following sentence:\n\"{sentence}\""
    model = "text-davinci-003"  # You can try different models based on your requirements

    # Make the API call
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )

    # Extract the rephrased sentence from the API response
    rephrased_sentence = response['choices'][0]['text'].strip()

    return rephrased_sentence

# Get input from clipboard
input_sentence = copy_clipboard()

# Call the rephrase_sentence function
rephrased_result = rephrase_sentence(input_sentence)

# Display the results
print(f"\nOriginal Sentence: {input_sentence}")
print(f"Rephrased Sentence: {rephrased_result}")

# Copy the rephrased sentence to the clipboard
pya.typewrite(rephrased_result)
# Notify the user that the rephrased sentence has been copied to the clipboard
print("Rephrased sentence has been copied to the clipboard.")


import os
import openai

API_KEY = os.environ.get("DAVINCI", "NO_API_KEY")
openai.api_key = API_KEY

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

def Prompt(prompt_msg):
    '''
    This function will take the prompt message and return a response from the AI
    param prompt_msg: The message to be sent to the AI
    '''
    response = "Sorry unfortunately I am not able to answer your question right now." 
    # Create a new prompt by concatenating the previous prompt with the new user input
    try:
      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_msg,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
      )
      print(response)

      return response.choices[0].text.strip()
    except Exception as e:
      print(e)
      return response
    
def ImagePrompt(prompt_msg):
    '''
    This function will take the prompt message and return a image response from the AI
    param prompt_msg: The message to be sent to the AI
    '''
    # Image generation
    try:
      response = openai.Image.create(
        prompt=prompt_msg,
        n=1,
        size="1024x1024"
      )
      image_url = response['data'][0]['url']
      return image_url
    except Exception as e:
      print(f"Davinci failed to generate image response: {e}")
      return "/noimage_bot_default.jpg"

    

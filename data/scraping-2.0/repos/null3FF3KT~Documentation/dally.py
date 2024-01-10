import openai
openai.api_key = "<Replace with your own OpenAI API key>"

def prompt_user():
  """TODO: parse user input for parameters like aspect resolution, and number of images(max 8)"""  
  """Prompts the user to enter a value and returns it."""
  prompt = "What image would you like to create: "
  value = input(prompt)
  return value


def main():
  while True:  
    value = prompt_user()
    """TODO: Needs error Handling""" 
    response = openai.Image.create(
      prompt=value,
      n=1,
      size="1024x1024"
    )

    for x in response['data']:
      print(x['url'])

    if (input("'q' to quit: ") == 'q'):
      break

if __name__ == "__main__":
  main()


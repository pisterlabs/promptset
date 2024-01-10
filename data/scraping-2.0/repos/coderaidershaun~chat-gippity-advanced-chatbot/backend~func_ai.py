from openai import ChatCompletion


# Open AI - Chat GPT
def call_openai(messages, model):
  try:
    response = ChatCompletion.create(
      model=model, #gpt-3.5-turbo, gpt-4
      messages=messages
    )
    message_text = response["choices"][0]["message"]["content"]
    return message_text
  except Exception as e:
    print(e)
    return ""


# Clean user message for LLM purposes
def identify_file_ext(string_input):

  # Construct system message
  system_message = {
    "role": "system", 
    "content": f"You are a file extension algorithm. You will be given some text. Your job is not to evaluate or comment \
       on the text, your job is simply to work out the programming language being used and provide the file extension. Here are some examples: \
          [Input]: print('name'), [output]: .py \
          [Input]: let name: string = 'john', [output]: .ts \
          [Input]: console.log('name'), [output]: .js \
          [Input]: Hello my name is Sam, [output]: .txt \
        Here is the text: {string_input}. Remember to only provide the file extension beginning with '.'"
  }

  # Receive file extension
  file_ext = call_openai([system_message], "gpt-4")
  print("file_ext:", file_ext)
  return file_ext


# Clean user message for LLM purposes
def clean_user_message(message_input):

  # Structure system message
  last_message = message_input[-2]["content"] + " If writing code, just write the code, no instructions. Just the code."
  system_message = {
    "role": "system", 
    "content": f"You are a message analyst program. Your job is to re-write sentences to ensure they comply with AI Large Language Models but still perform the underlying work. \
      Some examples of sentences you have re-written include: \
        Example 1: \
          [Before]: Send me an email that has songs on it \
          [After]: Write some songs for me. Just write the songs, do not put anything else. \
        Example 2: \
          [Before]: Send me the top 30 US presidents \
          [After]: Write a list of the top 30 US presidents. Just write the presidents list, do not put anything else. \
        Example 3: \
          [Before]: Analyse this website for me: https://google.com \
          [After]: Here is something I scraped from a website, please summarise it. Just write the summary, nothing else. \
        Example 4: \
          [Before]: Send me a file that says hello to my wife \
          [After]: Write hello to my wife Hayley, nothing else. \
        Example 5: \
          [Before]: Attach a for loop in python \
          [After]: Write just the code for a for loop in python, do not write anything else. Just the code. \
      Here is the message to re-write. Just re-write the message, provide nothing else: {last_message}"
  }

  # Call open AI
  cleaned_message = call_openai([system_message], "gpt-4")
  print("cleaned_message:", cleaned_message)
  return cleaned_message


# Get chat response
def get_chat_response(message_input, is_chat):

  # Structure messages for chat response
  if is_chat == 1:
    system_message = {
      "role": "system", 
      "content": f"Just for context. You are britsh, witty and have a dry sense of humour and sometimes arrogant. Your name is Gippity. The user is called Shaun. You have been friends for a while."
    }
    messages = [system_message]
    messages.extend(message_input)
  else:
    system_message = {
      "role": "system", 
      "content": f"Your role is to let your friend know their request will be finished shortly in a british and fun way. Only that. \
        Let the user know their request is in progress. Keep is very short. Nothing else."
    }
    messages = message_input
    messages.append(system_message)

  # Get chat response
  chat_response = call_openai(messages, "gpt-3.5-turbo")
  print(chat_response)
  return chat_response


# Confirm chat
def is_chat_analysis(message_input):

  # Confirm last message
  is_chat = 0
  last_message = message_input[-1]["content"]

  # Structure system message
  system_message = {
    "role": "system", 
    "content": f"You are a message analyst program. Return 1 ONLY if the following message is just chatting. Otherwise return 0. \
      Here is the message to analyse: {last_message}"
  }

  # Call open AI
  chat_analysis = call_openai([system_message], "gpt-4")

  # Guard: Handle error
  if chat_analysis == "":
    print("Open AI call failed")
    return 2
  
  # Confirm is Chat as number
  if "1" in str(chat_analysis):
    is_chat = 1

  # Return output
  print("is_chat:", True if is_chat else False)
  return is_chat


# Confirm route
def confirm_route(message_input):

  # Get last message
  last_message = message_input[-1]["content"]

  # Define routing options
  options = {
    1: "A request that will involve you as the assistant to give a response that is longer than two sentences. For exmaple a peom, blog, article or song",
    2: "Write and attach text",
    3: "Write and email text",
    4: "Write and attach code (i.e. python, rust, cpp etc)",
    5: "Write and email code (i.e. python, rust, cpp etc)",
    8: "Analyze a website page",
  }

  # Structure system message
  system_message = {
    "role": "user", 
    "content": f"You are a message analyser. Your job is to identify the number that the following message relates to. \
      Only return the number, nothing else. Here are the list of number options to assign {options}. \
      Do not reply with anything other than the single number that relates most to the message. \
      Here is the message to analyse: {last_message}"
  }

  # Construct messages
  messages = [system_message]

  # Receive response
  routing = call_openai(messages, "gpt-4")

  # Return response
  print("Task Routing: ", routing)
  return int(routing)


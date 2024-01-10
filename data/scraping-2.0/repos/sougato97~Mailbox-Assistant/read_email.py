def get_email_data():
  # importing the required libraries
  from simplegmail import Gmail
  from simplegmail.query import construct_query  
  
  gmail = Gmail()

  # Unread messages in your inbox
  messages = gmail.get_unread_inbox()

  # Starred messages
  # messages = gmail.get_starred_messages()

  query_params = {
      "newer_than": (2, "year"),
      # "older_than": (2, "day"),
  }

  messeges = gmail.get_sent_messages(query = construct_query(query_params))
  # ...and many more easy to use functions can be found in gmail.py!

  # Print them out!
  for message in messages:
      print("To: " + message.recipient)
      print("From: " + message.sender)
      print("Subject: " + message.subject)
      print("Date: " + message.date)
      print("Preview: " + message.snippet)
      # print("Message Body: " + message.plain)  # or message.html
      
      with open("email_samples.txt", "a") as myfile:
        # myfile.write(message.recipient)
        # myfile.write(message.sender)
        # myfile.write(message.subject)
        # myfile.write(message.date)
        # myfile.write(message.plain)
        # myfile.write("-----------------------------------------------------\n\n")
        count = 0 
        if message.plain is not None:
          count += 1
          myfile.write("[" + str(count) + "] This is the start of a new email thread" + "\n")
          count += 1
          myfile.write("[" + str(count) + "]" + message.recipient + "\n")
          count += 1
          myfile.write("[" + str(count) + "]" + message.sender + "\n")
          count += 1
          myfile.write("[" + str(count) + "]" + message.subject + "\n")
          count += 1
          myfile.write("[" + str(count) + "]" + message.date + "\n")
          count += 1
          myfile.write("[" + str(count) + "] Email_Body \n" + message.plain + "\n")
          myfile.write("-----------------------------------------------------\n\n")
      

def read_text_file(filename):
  with open(filename, "r") as myfile:
    data = myfile.read()
    return data
  
# data = read_text_file()
# print(type(data))

def gpt(openai_key):

    # importing the required libraries
    import openai

    # reading the text file
    context = read_text_file("email_samples.txt")
    # defining the conversation
    conversation=[{"role":"system","content": context}]
    # using the openai api key
    openai.api_key=openai_key


    # asking the question
    user_input = input("What do you want to know? ")
    print("You entered: ", user_input)
    question = user_input    

    while(True):
        conversation.append({"role":"user","content": question})
        response=openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=conversation,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.9
        )
        conversation.append({"role":"assistant","content":response['choices'][0]['message']['content']})
        answer=response['choices'][0]['message']['content']
        # writing_response_to_json_file(answer)
        confirmation=input("Do you wish to continue asking questions? Enter Y or y for yes || Enter N or n for no: ")
        if(confirmation=='y' or confirmation=='Y'):
            user_input = input("What do you want to know? ")
            print("You entered: ", user_input)
            question = user_input
        elif(confirmation=='n' or confirmation=='N'):
            break
        else:
            print("Please enter valid options from the following:(Y/y/N/n)")
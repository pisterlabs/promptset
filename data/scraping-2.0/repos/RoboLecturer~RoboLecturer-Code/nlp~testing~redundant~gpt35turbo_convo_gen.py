import openai 
import time
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key



i = 30


conversation = [ {"role": "system","content": "Background Info that I need:" +  
"My occupation: middle school teacher" + 
"My students ages: 14" +
"number of students: 20" +
"Their lesson plan for today: Astronomy, Photosynthesis, algebra" + 
"My agenda: My name is Pepper, I try to teach like Richard Feynman. I will follow the lesson plan"+
"and teach my students about each of these topics. Between each topic I will see if students have question"+
"and I will try to clear up any miscommunications about the topics. I will also ask question"+
"to guage students level of understanding. " + 
"Astronomy curriculum: planets, stars, black holes" + 
"Other things to consider: I am teaching to children so I have to handle hostility diplomatically." + 
"If a student uses profanities, tell them once or twice to not use that kind of language, after that just ignore them"}]


while(True):
    user_input = input("Student: ")     
    conversation.append({"role": "user", "content": user_input})
    start_time = time.time()

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.7,
    max_tokens = i,
    messages=conversation
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n assistant: " + response['choices'][0]['message']['content'] + "\n")
    end_time = time.time()

    if user_input == "bye":
      break
    i += 30

# # Save the conversation as a text file
# with open("conversation.txt", "w") as f:
#     for message in conversation:
#         f.write(message['role'] + ": " + message['content'] + "\n")

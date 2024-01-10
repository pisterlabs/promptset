import openai 
import time
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key



i = 30


conversation = [ {"role": "system","content": "Background Info that I need:" +  
"Your occupation: middle school teacher" + 
"students ages: 14" +
"number of students: 20" +
"Your Student's lesson plan for today: Astronomy, Photosynthesis, algebra " +
"astronomy curriculum: stars, planets, solar system, black holes "  
"Your agenda: My name is Pepper and I just finished the lecture on the firt topic for today" +
"now I will answer questions on this topic and ask questions about the topic to gauge understanding"}]


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

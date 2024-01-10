import banana_dev as banana
import openai

openai.api_key = "YOUR_API_KEY_HERE" # get this from OpenAI
carrot_api_key = "YOUR_API_KEY_HERE" # get this from banana.dev

def ask_next_question(conversation_history):
  """
  enter the existing conversation, ending with Beth's response + 'Albert:'
  returns the next question Albert asks
  """
  response = openai.Completion.create(
    engine="text-davinci-001",
    prompt="Two friends are playing a game. Albert is trying to deeply understand the image Beth is looking at, without seeing the image. Albert can ask Beth any questions he wants, but he can't see the image.\n\nPicture #1\n\nAlbert: What is this?\nBeth: This is a baseball game.\nAlbert: How many people are on the field?\nBeth: There are two people on the field.\nAlbert: What are the two people on the field doing?\nBeth: One person an umpire and the other is pitching.\nAlbert: Is the pitcher in the middle of throwing a ball?\nBeth: The pitcher is in the middle of throwing a ball.\nAlbert: What color uniform is the pitcher wearing?\nBeth: The pitcher is wearing a white uniform.\nAlbert: Are there a lot of people in the stands?\nBeth: There are many people in the stands.\nAlbert: Is the pitcher right-handed or left-handed?\nBeth: The pitcher is right-handed.\nAlbert: Where is the umpire in relation to the pitcher?\nBeth: The umpire is behind the pitcher.\n\n###\n\nPicture #2\n\n{}\nAlbert:".format(conversation_history),
    temperature=0.5,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["Albert", "Beth", "###", "Picture #3"]
  )

  return response['choices'][0]['text'].strip()

def remove_mention_of_image_from_question(question):
  """
  Converts the question to remove any mention of it being a picture, so as to not confuse carrot
  takes in a question
  returns a cleaned question
  """
  response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Convert the statement to remove any mention of it being a picture:\n\nBefore: How many people are on the field in the picture?\nAfter: How many people are on the field?\n\n###\n\nBefore: How many dancers are in the image?\nAfter: How many dancers are there?\n\n###\n\nBefore: {}\nAfter:".format(question.strip()),
  temperature=0.2,
  max_tokens=64,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["###", "Before", "After"]
  )

  return response['choices'][0]['text'].strip()

def call_carrot(text, imageURL):
  api_key = carrot_api_key
  model_key = "carrot"

  model_parameters = {
  "text": text,
  "imageURL": imageURL
  }

  out = banana.run(api_key, model_key, model_parameters)
  return out["modelOutputs"][0]["answer"]

def carrot_answer_to_response(question, answer):
  """
  given a question and it's carrot-generated short answer, generate a sentence to respond to the question
  returns a sentence
  """
  response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Given a question and a correct answer, write a well-formed response to the question:\n\nQuestion: What is this image?\nCorrect Answer: baseball game\n\nWell-formed Response: This is a baseball game.\n\n###\n\nQuestion: How many players are in the image?\nCorrect Answer: 2\n\nWell-formed Response: There are two players in this image.\n\n###\n\nQuestion: {}\nCorrect Answer: {}\n\nWell-formed Response:".format(question.strip(), answer.strip()),
  temperature=0.7,
  max_tokens=64,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["###", "Question", "Correct", "Well-formed"]
  )

  return response['choices'][0]['text'].strip()

def analyze_image(url):
  # first question to seed the process
  initial_question = "What is this?"
  initial_carrot_answer = call_carrot(initial_question, url)
  initial_response = carrot_answer_to_response(initial_question, initial_carrot_answer)
  conversation_history = "Albert: {}\nBeth: {}".format(initial_question, initial_response)
  
  # continue asking 8 more questions and getting answers
  for i in range(8):
    next_question = ask_next_question(conversation_history)
    next_question = remove_mention_of_image_from_question(next_question)
    next_carrot_answer = call_carrot(next_question, url)
    next_response = carrot_answer_to_response(next_question, next_carrot_answer)
    conversation_history = "{}\nAlbert: {}\nBeth: {}".format(conversation_history, next_question, next_response)
  
  # extract all of Beth's answers and save them in a list
  answers = []
  for line in conversation_history.split("\n"):
    if "Beth" in line:
      answers.append(line.split(":")[1].strip())
  
  return answers, conversation_history

# do this a few times and check the avg to improve accuracy?
def fact_check(answers, imageURL):
  final_statements = []
  for answer in answers:
    response = openai.Completion.create(
    engine="text-davinci-001",
    prompt="For each statement, turn it around into a question:\n\nStatement: The game is being played in the afternoon.\nQuestion: Is it the afternoon?\n\n###\n\nStatement: The players are on a baseball field.\nQuestion: Are the players on a baseball field?\n\n###\n\nStatement: {}\nQuestion:".format(answer),
    temperature=0.26,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["###", "Statement", "Question"]
    )

    question = response['choices'][0]['text'].strip()
    
    carrot_answer = call_carrot(question, imageURL)
  

  
    if 'y' in carrot_answer:
      final_statements.append(answer)

  return final_statements

def meta_analyze_image(url):
  all_answers = []
  for i in range(5):
    answers, conversation_history = analyze_image(url)
    all_answers = all_answers + answers
  
  all_answers = list(set(all_answers))

  checked_answers = fact_check(all_answers, url)

  checked_answers = "\n".join(checked_answers)

  response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Given a set of statements about an image, describe the image:\n\nStatements:\nThe game is being played in the afternoon.\nThe people in this image are playing baseball.\nThe pitcher throws with their right hand.\nThe players are on a baseball field.\nThe score is 0-0.\nYes, the game is being played.\nThis is a baseball game.\nThis is a pitcher.\nThe pitcher's arm is throwing the ball.\nThe pitcher is facing the catcher in this image.\nThe pitcher's hat is black.\nThe uniforms are white.\nThe people in the front row are standing.\nThe ball is in the hand.\nThe pitcher is throwing the ball.\nThe people in the back row stand.\nThe pitcher's uniform is white.\nThe pitcher is on the pitcher's mound.\nThe background is the bleachers.\n\nDescribe the image:\nThis image is a photograph of a baseball game, in the afternoon. In the background, there are bleachers, and in the back and front rows, people are standing. The pitcher, who is wearing a white uniform and has a black hat, is on the pitcher's mound and is throwing the ball to the catcher with their right hand. The catcher is also in white, and is in front of the pitcher. The score of the game is 0-0.\n\n###\n\nStatements:\n{}\n\nDescribe the image:".format(checked_answers),
  temperature=0.7,
  max_tokens=200,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
  )
  
  return response['choices'][0]['text'].strip()

meta_analyze_image("https://www.si.com/.image/t_share/MTY4MTk3MTQ1NjcyMzYxODU3/tennis-inlinejpg.jpg")

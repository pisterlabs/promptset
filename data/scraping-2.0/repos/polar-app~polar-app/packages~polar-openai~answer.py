import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

questions = [
    "Who did Tutankhamun marry?",
    "How old was Tutankhamun when he rose to the throne?",
    "Who restored the Ancient Egyptian religion?",
    "Where did King Tut move his father's remains?",
    "Who funded Howard Carter's discovery of Tutankhamun's tomb?",
    "What was the status of King Tut's tomb when it was found?",
    "What is the curse of the pharaohs?",
    "How tall was King Tut?",
    "How tall was King Tut in feet?",
    "How tall was King Tut in cm?",
    "How did King Tut die?",
]

# models = ["curie", "davinci"]
# models = ["davinci"]
models = ["davinci"]
#models = ["davinci"]
max_tokens=35
search_model='curie'

def doAnswer(model, question):

    response = openai.Answer.create(
      search_model=search_model,
      model=model,
      question=question,
      file="file-BW3Opoe0JbJJzto76qSn7wOp",
      examples_context="In 2017, U.S. life expectancy was 78.6 years.",
      examples=[
        ["What is human life expectancy in the United States?","78 years."]
      ],
      max_tokens=max_tokens,
      stop=["\n", "<|endoftext|>"],
    )

    # print(response)
    #print("%-10s %-70s %s" % (model, question, response.answers))
    print("%-70s\n   %s" % (question, response.answers[0]))

for model in models:
    print(model)
    print("max_tokens: %s" % max_tokens)
    print("search_model: %s" % search_model)
    print("====")
    for question in questions:
        doAnswer(model, question)

#
# response = openai.Answer.create(
#   search_model="davinci",
#   model="davinci",
#   question="How did King Tut die?",
#   file="file-BW3Opoe0JbJJzto76qSn7wOp",
#   examples_context="In 2017, U.S. life expectancy was 78.6 years.",
#   examples=[
#     ["What is human life expectancy in the United States?","78 years."]
#   ],
#   max_tokens=25,
#   stop=["\n", "<|endoftext|>"],
# )
#
# print(response.answers)
#
# print("====")
# print(response)

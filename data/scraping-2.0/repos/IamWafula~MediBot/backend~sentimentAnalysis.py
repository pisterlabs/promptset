import openai
api_key = "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"
openai.api_key = api_key
import json
#import nltk
#nltk.download('punkt')

def checkSentiment(message):

  currentIdeaState = {
    'Sentiment' : ''
  }
  prompt = '''
  def categorizeResponse(user):
    if userHasSymptom() = Positive:
      return('Positive')
    elif not userHasSymptom() = Negative:
      return('Negative')
    elif userIsUncertain() = Neutral:
      return('Neutral')

    Examples::
    Q: Yes, I feel pain
    A: Positive
    Q: I don't feel pain.
    A: Negative
    Q: My day has been 👍
    A: Positive
    Q: I am not sure
    A: Neutral
    Q: I feel bad
    A: Positive
    Q: I do not feel bad
    A: Negative
    Q: I don't think I have a headache
    A: Neutral
  

    Q: This new music video blew my mind
    A: 

  #Inputs
  current_state = {state}
  User_message = {message}

  #Output
  new_state = 
  '''.format(message=message, state=json.dumps(currentIdeaState))

  response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.5,
  max_tokens=500,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
  responseText = response.choices[0].text
  #print(type(responseText))
  
  '''sentences = nltk.sent_tokenize(responseText)
  outputList = []
  for a in sentences:
    a = a.replace("\"Clarifying question\"", "").replace(":","").replace("\"", "")
    if "?" in a:
      outputList.append(a)'''

  return json.loads(responseText)



# sentimentResult = checkSentiment(sentimentState, userInput)

# print(sentimentResult)
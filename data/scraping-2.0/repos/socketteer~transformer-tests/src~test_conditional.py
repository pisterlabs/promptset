import openai
from conditional import event_probs

trivia_question = """Q: What is the largest country in the world?
A:"""

choices = ['Russia', 'Canada', 'China', 'United States']


probs, normal_probs = event_probs(trivia_question, choices, engine='davinci')
print('probs: ', probs)
print('normalized probs: ', normal_probs)

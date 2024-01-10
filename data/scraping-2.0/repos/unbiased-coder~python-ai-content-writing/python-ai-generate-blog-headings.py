from openai_helper import openai_create_outline

headings = 'The history of Tesla\'s inventions'
print ('Generating headings for: ', headings)
print(openai_create_outline(headings))




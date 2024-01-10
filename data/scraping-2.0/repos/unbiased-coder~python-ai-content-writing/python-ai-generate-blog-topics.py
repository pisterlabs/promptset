from openai_helper import openai_blog_ideas

keywords = 'Nichola Teslas Inventions'
print ('Generating ideas for: ', keywords)
print(openai_blog_ideas(keywords))

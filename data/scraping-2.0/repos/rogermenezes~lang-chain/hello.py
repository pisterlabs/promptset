from langchain.llms import OpenAI

# testing out a comment
llm = OpenAI(model="text-davinci-003", temperature=0.9)
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))
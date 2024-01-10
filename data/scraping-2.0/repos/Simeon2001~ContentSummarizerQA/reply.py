import openai

openai.api_key = "Put Your Openai Key"

messages = []
messages.append( {
    "role": "system", "content": "as an AI assistant that empowers learners to create comprehensive learning materials and assessments with ease. generate variety of question formats from the learning materials which are: multiple-choice, transformation exercises, gap-filling tasks, matching questions, close tests, true/false questions, open questions, and error correction exercises."
})

def summary(msg):
	messages = []
	messages.append( {
    "role": "system", "content": "as an AI assistant that empowers learners to create comprehensive learning materials and assessments with ease. summarize learning materials for quick review"
	})
#    try:
	messages.append(
    {
        "role": "user", "content": msg
    }
	)
	completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = messages
	)

	response = (completion.choices[0].message)
	messages.append({
    "role": "assistant", "content": response["content"]})
	return response["content"]

def answer(msg):
#    try:
	messages.append(
    {
        "role": "user", "content": msg
    }
	)
	completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = messages
	)

	response = (completion.choices[0].message)
	messages.append({
    "role": "assistant", "content": response["content"]})
	return response["content"]

#    except:
#       msg = "my brain is currently snoozing"
#        return msg
with open('question.txt') as f:
	lines = f.readlines()
question = lines

# Generate Multiple Question
print(answer(question[0]))

# Summarize Learning Materials
print(summary(question[0]))
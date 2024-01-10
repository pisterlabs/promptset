import json
import random
import string
import openai
from tqdm import tqdm

# filter transcripts to find transcripts that are debating a *specific* bill
# start by loading the transcripts


with open("./daterangegovinfo02.json") as f:
	data = json.load(f)

# pick a random 1000 transcripts to work with
# seed the randomness with a fixed value so that we can reproduce the results


random.seed(0)

previous_run_transcripts = random.sample(data, 300)

random.seed(0)

transcripts = random.sample(data, 1000)

# check that the first 300 transcripts are the same in both lists

for i in range(300):
	if previous_run_transcripts[i] != transcripts[i]:
		print("ERROR: random samples are not equal")
		print("previous_run_transcripts[i]: " + str(previous_run_transcripts[i]))
		print("transcripts[i]: " + str(transcripts[i]))
		print("i: " + str(i))
		exit(1)

# fix whitespace in transcripts
# replace \n with newlines
for t in transcripts:
	t['transcript'] = t['transcript'].replace("\\n", "\n")

# filter out transcripts that are less than 800 characters long (likely not a debate)

transcripts = [t for t in transcripts if len(t['transcript']) > 800]

# filter out transcripts that are less than 1500 characters long and contain the phrase
# "Congress has the power to enact this legislation pursuant to the following:"
# (ignore whitespace)

no_whitespace = {ord(c): None for c in string.whitespace}

transcripts = [t for t in transcripts if len(t['transcript']) > 1500 or
			   "Congress has the power to enact this legislation pursuant to the following:".translate(no_whitespace)
			   not in t['transcript'].translate(no_whitespace)]

# while True:
# 	print("\n" * 100)
# 	item = random.choice(data)
#
# 	# replace \n with newlines
# 	transcript = item['transcript'].replace("\\n", "\n")
# 	print(transcript)
# 	input("Press enter for another transcript, or ctrl+c to exit")

# ***********************************************************************************

# Now we ship these off to the GPT-3 instruct api to see if it's a debate about a bill
# we don't ask *which* bill @ this point, just if it's a debate about a bill


client = openai.OpenAI(api_key="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")

base_prompt = "This is an excerpt from a transcript from the United States Congress. Is this a debate about a bill? Answer \"TRUE\" or \"FALSE\".\n"


def get_answer(transcript, model):
	response = client.chat.completions.create(
		model=model,
		messages=[
			{
				"role": "system",
				"content": base_prompt
			},
			{"role": "user", "content": t['transcript']}
		],
	)
	answer = response.choices[0].message.content
	if answer != "TRUE" and answer != "FALSE":
		print("ERROR: unexpected answer from GPT: " + answer)
		return None
	return answer == "TRUE"


continue_from = 300

for t in tqdm(transcripts[continue_from:]):
	try:
		answer = get_answer(t['transcript'], "gpt-3.5-turbo-16k")
	except:
		try:
			answer = get_answer(t['transcript'], "gpt-4-1106-preview")
		except:
			answer = None

	if answer is None:
		with open("debate_transcripts_unknown.json", "a") as f:
			json.dump(t, f)
			f.write("\n")

	if answer:
		# print(t['transcript'])
		# input("Press enter for another transcript, or ctrl+c to exit")
		with open("debate_transcripts.json", "a") as f:
			json.dump(t, f)
			f.write("\n")

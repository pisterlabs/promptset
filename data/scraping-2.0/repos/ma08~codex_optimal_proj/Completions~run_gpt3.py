import os
import sys
import openai

NEW_DIR = "questions-w-gpt3-context"
Q_FILE = "question.txt"
best_fname = "q_w_best_ctxt"
best_query = "\nWhat's the best way to solve this problem?"
optml_fname = "q_w_optml_ctxt"
optml_query = "\nWhat is the optimal solution to this problem?"
optml_stps_fname = "q_w_optml_stps_ctxt"
optml_steps_query = "\nHere are the steps to getting the optimal solution to this problem:\n1."
GPT_CTXT_QUERIES = {
	best_fname: best_query,
	optml_fname: optml_query,
	optml_stps_fname: optml_steps_query
}
ENGINE = "text-davinci-002"
TEMPERATURE = 0 #T
MAX_TOKENS=256

# Change accordingly...
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_gpt3(input_dir):
	path = os.path.join(input_dir, NEW_DIR)
	try:
		os.mkdir(path)
	except OSError as error:
		print(error)

	infile = os.path.join(input_dir, Q_FILE)
	orig_q = open(infile, 'r').read()
	for fname, query in GPT_CTXT_QUERIES.items():
		prefix = "-----PROBLEM-----\n\n"
		input_prompt = f"\"\"\"\n{prefix}{orig_q}\n\"\"\"{query}"

		response = openai.Completion.create(
		  engine=ENGINE,
		  prompt=input_prompt,
		  temperature=TEMPERATURE,
		  max_tokens=MAX_TOKENS,
		  top_p=1,
		  frequency_penalty=0,
		  presence_penalty=0
		)

		new_question = ""
		suffix = "\n\n-----HINTS-----\n\n"
		if query == optml_steps_query:
			new_question = orig_q + suffix + query + response["choices"][0]
		else:
			new_question = orig_q + suffix + response["choices"][0]

		out = os.path.join(path, fname + ".txt")
		with open(out, 'w') as wf:
			wf.write(new_question)


if __name__ == "__main__":
	input_dir = sys.argv[1] # ./test/intro-questions.txt_dir/4000/
	run_gpt3(input_dir)

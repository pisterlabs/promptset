import openai
import time
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

def generate():
	line = "How are stars born"
	openai.api_key = openai_API_key

	completions = openai.Completion.create(
		engine = "text-curie-001",
		prompt = "explain the following point in a single paragraph in the style of an excited lecturer: {line}",
		max_tokens = 1024,
		n = 1,
		temperature = 0.2,
	)
	
	response = completions.choices[0]["text"]
	return response

num_iters = 20
sum = 0

for i in range(num_iters):
	start_time = time.time()
	response = generate()
	end_time = time.time()
	sum += end_time - start_time

av_time = sum/num_iters
print(response)
print(f"Average execution time: {av_time}")


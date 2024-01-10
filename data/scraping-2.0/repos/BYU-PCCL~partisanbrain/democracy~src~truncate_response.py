import openai

def truncate_response(response):
	response = response.choices[0].text
	response = response.split('\n')[0]
	response = response.split('.')[:-1]
	return response

print(response)
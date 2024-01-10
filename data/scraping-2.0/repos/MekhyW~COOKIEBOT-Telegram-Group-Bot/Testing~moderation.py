import openai
openai.api_key = ''

text = input("Texto: ")
response = openai.Moderation.create(input=text)
results = response['results'][0]['category_scores']

print(response['results'][0])
if any(x > 0.2 for x in [results['hate'], results['hate/threatening'], results['self-harm'], results['self-harm/instructions'], results['self-harm/intent'], results['sexual/minors'], results['violence/graphic']]):
    print("Censored")
else:
    print("Not censored")
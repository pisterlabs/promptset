import cohere

prompt = "Documents provided to the House Select Committee investigating the January 6, 2021, insurrection by the US Secret Service show that the agency and its law enforcement partners were aware of social media posts that contained violent language and threats aimed at lawmakers prior to the US Capitol attack."
print(prompt)
co = cohere.Client('JgMx33cwKF2GpRjKMh2Xc6BUen2CL1gfyfu3Zpw7')
response = co.generate(
  model='xlarge',
  prompt=prompt,
  max_tokens=50,
  temperature=0.5,
  num_generations=5,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')


print('Prediction: {}'.format(response.generations[0].text))



print("amount summarized:" + str(len(response.generations[0].text)/len(prompt)))
from privtoken import OPEN_AI_KEY
import fetch_data as fd
import openai
openai.api_key = OPEN_AI_KEY

raw_data = fd.get_raw_financials('TSLA', 5)
# whitelist keys that contain the word "research"
revenue_keys = [key for key in raw_data[0].keys() if "research" in key.lower()]

#generate davinci 3 prompt from the whitelist keys
prompt = "From the following list of keys, return the key that refers to the company's research and development expenses: "
for key in revenue_keys:
    prompt += f"{key}, "

print(prompt)

# generate a response from davinci
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1024,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# The response from davinci 3 is the name of the key that refers to the company's research and development expenses
# Use the response to get the value from the raw_data
response_text = response["choices"][0]["text"]
response_text = response_text.replace("\n", "")
for year in range(0, len(raw_data)):
    print(raw_data[year][response_text])

# use LLM to classify reddit posts based on the degree of illness: severe illness, light illness and no illness

# this program is designed so that the occurence of exceeding token per minute error can be elimitated

# the program will try to use the faster gpt-3.5-turbo, but automically switch to gpt-4 when the input tokens are too much. The output is specified to match a specific pattern

import openai

# Authenticate your API key
openai.api_key = "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"

labeled_examples = {
    "no illness": ["Thousands thronged a religious festival in Sanand without masks and social distancing even as the Gujarat government has imposed a Mini Lockdown till May 12", "Just went for a run and feeling energized.", "Her tongue was cut, spine broken. What about you?", "TIL of Ghulam Dastagir, a Stationmaster who refused to leave his post during the Bhopal Gas Tragedy & saved thousands of lives by not letting any trains stop at the station. He spent the next 2 decades in & out of hospital due to long exposure to the gas before passing in 2003"],
    "light illness": ["India Is Making It Nearly Impossible for Homeless People to Get Vaccinated. India’s vaccination program requires a mobile phone and a home address. Many people have neither.", "Slight headache and fatigue.", "Pizza delivery boy tests positive, 72 families in South Delhi ordered to quarantine themselves."],
    "severe illness": ["I'm in a lot of pain today.", "We've only been here a few hours and have seen half a dozen people die while they wait for treatment.", "My grandmother fought and beat COVID after battling it for a month, and turned 94 today.", "Corona Donors"]
}

i = 0
j = 10
Illness = []

while (j < 1087):
  data2 = data.iloc[i:j]
  content_list = data2['tweet'].values.tolist()
  content_list2 = data2['hashtags'].values.tolist()
  for k in range(len(content_list)):
    if str(content_list2[k]) != '[]':
        batch_text = content_list[k] + '. The hashtags of the post is enclosed in the following list: ' + str(content_list2[k])
    else:
        batch_text = content_list[k]
    try:
      response = openai.ChatCompletion.create(
          model="gpt-3.5_turbo",
          messages=[
                {"role": "system", "content": "Classify each of the following social media posts as indicating no illness, light illness, or severe illness. Only output 'no illness', 'light illness', or 'severe illness'. Do not output any word other than those three. Also, we define illness as 'infectious, causing harm to individual's health or the functioning of community. Remember that violence, natural accidents do not count"},
                {"role": "system", "content": f"Here are some examples: {labeled_examples}"},
                {"role": "user", "content": batch_text}
            ],
           max_tokens=50,
           n=1,
           stop=None,temperature=0.3,
          )
    except:
      response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
                {"role": "system", "content": "Classify each of the following social media posts as indicating no illness, light illness, or severe illness. Only output 'no illness', 'light illness', or 'severe illness'. Do not output any word other than those three. Also, we define illness as 'infectious, causing harm to individual's health or the functioning of community. Remember that violence, natural accidents do not count"},
                {"role": "system", "content": f"Here are some examples: {labeled_examples}"},
                {"role": "user", "content": batch_text}
            ],
           max_tokens=50,
           n=1,
           stop=None,
           temperature=0.3,
          )
    re = response.choices[0].message.content
    re = re.lower()
    Illness.append(re)
    i = len(Illness)
    j = len(Illness) + 10
  print(len(Illness))

# classify the last few rows 

data2 = data.iloc[i: 1087]
content_list = data2['tweet'].values.tolist()
content_list2 = data2['hashtags'].values.tolist()
  for k in range(len(content_list)):
    if str(content_list2[k]) != '[]':
        batch_text = content_list[k] + '. The hashtags of the post is enclosed in the following list: ' + str(content_list2[k])
    else:
        batch_text = content_list[k]
    try:
      response = openai.ChatCompletion.create(
          model="gpt-3.5_turbo",
          messages=[
                {"role": "system", "content": "Classify each of the following social media posts as indicating no illness, light illness, or severe illness. Only output 'no illness', 'light illness', or 'severe illness'. Do not output any word other than those three. Also, we define illness as 'infectious, causing harm to individual's health or the functioning of community. Remember that violence, natural accidents do not count"},
                {"role": "system", "content": f"Here are some examples: {labeled_examples}"},
                {"role": "user", "content": batch_text}
            ],
           max_tokens=50,
           n=1,
           stop=None,temperature=0.3,
          )
    except:
      response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
                {"role": "system", "content": "Classify each of the following social media posts as indicating no illness, light illness, or severe illness. Only output 'no illness', 'light illness', or 'severe illness'. Do not output any word other than those three. Also, we define illness as 'infectious, causing harm to individual's health or the functioning of community. Remember that violence, natural accidents do not count"},
                {"role": "system", "content": f"Here are some examples: {labeled_examples}"},
                {"role": "user", "content": batch_text}
            ],
           max_tokens=50,
           n=1,
           stop=None,
           temperature=0.3,
          )
    re = response.choices[0].message.content
    re = re.lower()
    Illness.append(re)
    i = len(Illness)
    j = len(Illness) + 10
print(len(Illness))

data['illness'] = Illness
# show the rows where chatgpt gives outputs that do not match the previous pattern

data2 = data.loc[(data["illness"] != "no illness") & (data["illness"] != "severe illness") & (data["illness"] != "light illness") ]
data2

# change the rows which output does not match prespecified pattern accordingly

data["illness"] = data["illness"].replace(["no context", "inappropriate content"], ["no illness", "no illness"])
data.at[726, "illness"] = "no illness"
data.at[605, "illness"] = "no illness"
data.at[402, "illness"] = "no illness"
data100 = data.loc[(data["illness"] != "no illness") & (data["illness"] != "severe illness") & (data["illness"] != "light illness")]
data.to_csv("dataset_with_illness_2.csv", index = False, encoding = "UTF-8")



